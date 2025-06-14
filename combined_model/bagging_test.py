import pandas as pd
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
import random
import numpy as np
import itertools
import pickle # Import pickle for saving/loading checkpoints
import pyarrow.parquet as pq # Import pyarrow for Parquet file handling
from tqdm import tqdm # Import tqdm for progress bar

# Explicitly import functions from your combined_recs_testing.py file
from combined_recs_testing_2 import ( # Corrected import to combined_recs_testing_2
    _load_recommendation_data,
    _process_user_ratings_file,
    _split_user_data_for_testing,
    _get_relevant_books_from_validation,
    get_combined_recommendations,
    recommend_fraction_of_top_n,
    recommend_cb_filtered_by_cf,
    recommend_by_multiplying_scores,
    # recommend_hybrid_strategy_2_and_1, # This function is commented out in combined_recs_testing.py
    evaluate_recommendations_overlap
)

# 1. Define the path to your CSV file
csv_file_path = "../train_test_split/split_data_combined/cross_val_ratings.csv"

# Define directories and file paths
CHECKPOINT_DIR = "bagging_evaluation_results"
# Checkpoint for pre-computed base model results (file paths)
PRECOMPUTED_CHECKPOINT_FILENAME = os.path.join(CHECKPOINT_DIR, "precomputed_combined_dfs_checkpoint.pkl")
# Checkpoint for overall subset evaluation progress
SUBSET_PROGRESS_CHECKPOINT_FILENAME = os.path.join(CHECKPOINT_DIR, "subset_progress_checkpoint.pkl")

TEMP_DATA_DIR = os.path.join(CHECKPOINT_DIR, "temp_precomputed_data") # Directory for temporary parquet files
FINAL_RECS_DATA_DIR = os.path.join(CHECKPOINT_DIR, "final_recs_data") # Directory for final recommendations dataframes

# Define step markers for subset progress
STEP_INIT = -1 # Before any step for a subset has started
STEP_MODELS_LOADED_AND_COMBINED = 0 # After loading models and combining DFs for all runs in a subset
STEP_SCORES_AVERAGED = 1 # After averaging scores for the subset
STEP_FINAL_RECS_GENERATED = 2 # After generating final recommendations for the subset
STEP_FINAL_RECS_SAVED = 3 # After saving the final recommendations DataFrame to file
STEP_EVALUATION_COMPLETE = 4 # After evaluation and saving CSV for the subset


def save_checkpoint(data, filename):
    """Saves the given data to a file using pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved to {filename}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving checkpoint to {filename}: {e}")

def load_checkpoint(filename):
    """Loads checkpoint data from a file using pickle."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint loaded from {filename}")
            return data
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading checkpoint from {filename}: {e}. Starting fresh.")
            return {}
    return {} # Return empty dict if file doesn't exist

def save_intermediate_df_to_file(df: pd.DataFrame, param_key: tuple) -> str:
    """
    Saves a DataFrame to a unique Parquet file in the temporary directory.
    Returns the path to the saved file.
    """
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    # Create a unique filename from the parameter key
    filename = f"nf{param_key[0]}_rs{param_key[1]}_gw{param_key[2]}.parquet"
    filepath = os.path.join(TEMP_DATA_DIR, filename)
    
    try:
        df.to_parquet(filepath, index=False)
        return filepath
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving intermediate DataFrame to {filepath}: {e}")
        return "" # Return empty string or None on failure

def load_intermediate_df_from_file(filepath: str) -> pd.DataFrame | None:
    """
    Loads a DataFrame from a Parquet file.
    Returns the DataFrame or None if loading fails.
    """
    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading intermediate DataFrame from {filepath}: {e}")
            return None
    return None # File does not exist

def save_final_recs_df_to_file(df: pd.DataFrame, subset_name: str) -> str:
    """
    Saves the final recommendations DataFrame for a subset to a Parquet file.
    Returns the path to the saved file.
    """
    os.makedirs(FINAL_RECS_DATA_DIR, exist_ok=True)
    filename = f"final_recs_df_{subset_name}.parquet"
    filepath = os.path.join(FINAL_RECS_DATA_DIR, filename)
    try:
        df.to_parquet(filepath, index=False)
        return filepath
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving final recommendations DataFrame to {filepath}: {e}")
        return ""

def load_final_recs_df_from_file(filepath: str) -> pd.DataFrame | None:
    """
    Loads the final recommendations DataFrame from a Parquet file.
    Returns the DataFrame or None if loading fails.
    """
    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading final recommendations DataFrame from {filepath}: {e}")
            return None
    return None


def collect_combined_dfs_for_single_run(
    training_data_df: pd.DataFrame,
    loaded_data_tuple: tuple,
    new_user_regularization_strength: int
) -> pd.DataFrame:
    """
    Generates combined recommendation scores (CB and CF) for all unrated books
    for each user in the training data for a single run.
    Returns a single DataFrame containing all books and scores for all users from this run.
    """
    all_users_combined_dfs_for_run = []
    
    if training_data_df.empty:
        return pd.DataFrame()

    # Wrap the user iteration with tqdm for a progress bar
    for index, user_row in tqdm(
        training_data_df.iterrows(),
        total=len(training_data_df),
        desc="Collecting User Scores",
        unit="users",
        leave=False # Do not leave the bar on screen after loop finishes
    ):
        user_id = user_row['User_id']
        user_books_training = user_row['Book_Identifications_List']
        user_ratings_training = user_row['Ratings_List']

        combined_df_for_user = get_combined_recommendations(
            user_books=user_books_training,
            user_ratings=user_ratings_training,
            loaded_data_tuple=loaded_data_tuple,
            new_user_regularization_strength=new_user_regularization_strength
        )

        if combined_df_for_user is not None and not combined_df_for_user.empty:
            combined_df_for_user['User_id'] = user_id
            all_users_combined_dfs_for_run.append(combined_df_for_user)
    
    if all_users_combined_dfs_for_run:
        return pd.concat(all_users_combined_dfs_for_run, ignore_index=True)
    return pd.DataFrame()


def generate_recommendations_from_averaged_scores(
    averaged_combined_df: pd.DataFrame,
    output_limit_per_strategy: int = 60
) -> pd.DataFrame:
    """
    Generates final recommendation lists for each user using their averaged scores.
    This replaces the original generate_all_recommendations_for_users but uses a
    pre-averaged combined_df for each user.
    """
    all_user_recommendations = []

    print("\n--- Generating All Recommendations from Averaged Scores ---")

    unique_users = averaged_combined_df['User_id'].unique()

    if averaged_combined_df.empty:
        print("Averaged combined DataFrame is empty. Cannot generate recommendations.")
        return pd.DataFrame(columns=[
            'User_id',
            'Rec_CB_Only',
            'Rec_CF_Only',
            'Rec_CB_Filtered_by_CF',
            'Rec_By_Multiplying_Scores',
        ])

    # Wrap the user iteration with tqdm for a progress bar
    for user_id in tqdm(
        unique_users,
        total=len(unique_users),
        desc="Generating Final User Recs",
        unit="user",
        leave=False # Do not leave the bar on screen after loop finishes
    ):
        user_averaged_df = averaged_combined_df[averaged_combined_df['User_id'] == user_id].copy()
        user_averaged_df['Rated_by_User'] = user_averaged_df['Rated_by_User'].astype(bool)

        user_recs = {'User_id': user_id}

        if user_averaged_df.empty:
            # print(f"Skipping recommendations for user {user_id} due to empty averaged_df.") # Suppress for cleaner tqdm output
            user_recs['Rec_CB_Only'] = []
            user_recs['Rec_CF_Only'] = []
            user_recs['Rec_CB_Filtered_by_CF'] = []
            user_recs['Rec_By_Multiplying_Scores'] = []
        else:
            user_recs['Rec_CB_Only'] = recommend_fraction_of_top_n(
                user_averaged_df, output_limit=output_limit_per_strategy, cb_fraction=1.0
            )
            user_recs['Rec_CF_Only'] = recommend_fraction_of_top_n(
                user_averaged_df, output_limit=output_limit_per_strategy, cb_fraction=0.0
            )
            user_recs['Rec_CB_Filtered_by_CF'] = recommend_cb_filtered_by_cf(
                user_averaged_df, output_limit=output_limit_per_strategy
            )
            user_recs['Rec_By_Multiplying_Scores'] = recommend_by_multiplying_scores(
                user_averaged_df, output_limit=output_limit_per_strategy
            )
        
        all_user_recommendations.append(user_recs)

    recommendations_df = pd.DataFrame(all_user_recommendations)
    print("\n--- All User Recommendations Generated Successfully from Averaged Scores ---")
    return recommendations_df


# --- Main Execution Loop for Bagging (Averaged Book Scores) ---
if __name__ == "__main__":
    # Define your specific parameter combinations here
    all_parameter_combinations = [
        (400, 15, 1),
        (400, 20 , 1),
        (300, 10, 1),
        (300, 15, 1), 
        (300, 20, 1),
        (200, 10, 1), 
        (200, 15, 1)
    ]
    
    # Filter combinations by num_features
    combinations_400 = [p for p in all_parameter_combinations if p[0] == 400]
    combinations_300 = [p for p in all_parameter_combinations if p[0] == 300]
    combinations_200 = [p for p in all_parameter_combinations if p[0] == 200]

    # Generate subsets where each subset contains exactly one of each num_features tier
    subset_definitions = {}
    subset_counter = 0
    # Use itertools.product to get all combinations by picking one from each group
    for combo_400 in combinations_400:
        for combo_300 in combinations_300:
            for combo_200 in combinations_200:
                current_subset_combinations = [combo_400, combo_300, combo_200]
                current_subset_combinations.sort() 
                
                subset_name_parts = []
                for nf, rs, gw in current_subset_combinations:
                    subset_name_parts.append(f"NF{nf}_RS{rs}")
                subset_name = "_".join(subset_name_parts)
                
                if subset_name not in subset_definitions:
                    subset_definitions[subset_name] = current_subset_combinations
                    subset_counter += 1

    # ADDITION: Add a subset that bags ALL 7 models together
    all_models_subset_name = "All_7_Models_Bagged"
    subset_definitions[all_models_subset_name] = all_parameter_combinations 


    # Parameters for recommendation output and relevance threshold (fixed for all subsets)
    output_limit_per_strategy = 60
    relevance_threshold = 3.0

    # 0. Initial Data Loading and Splitting (Done once, as the user ratings are static)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initial data processing (one-time step)...")
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            user_data_processed_df = _process_user_ratings_file(csv_file_path)
            training_df, validation_df = _split_user_data_for_testing(
                user_data_processed_df, validation_split_ratio=0.5
            )
            relevant_val_books_df = _get_relevant_books_from_validation(
                validation_df, relevance_threshold=relevance_threshold
            )
    
    if training_df.empty or validation_df.empty or relevant_val_books_df.empty:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Critical data (training, validation, or relevant books) is empty. Exiting.")
        sys.exit(1)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initial data processing complete.")

    # --- Optimization: Pre-compute combined_dfs for all base parameter combinations ---
    # Attempt to load checkpoint, otherwise start with an empty dictionary
    precomputed_combined_dfs = load_checkpoint(PRECOMPUTED_CHECKPOINT_FILENAME)
    if precomputed_combined_dfs:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resuming pre-computation from checkpoint. Already processed {len(precomputed_combined_dfs)} configurations.")

    print(f"\n--- Pre-computing combined scores for all {len(all_parameter_combinations)} base model configurations ---")
    for i, (nf, rs, gw) in tqdm(
        enumerate(all_parameter_combinations),
        total=len(all_parameter_combinations),
        desc="Pre-computing Base Models",
        unit="model_config",
        leave=False # Do not leave the bar on screen after loop finishes
    ):
        param_key = (nf, rs, gw) # Use the tuple as a key

        # Skip if this configuration's data is already in the checkpoint AND the file exists
        if param_key in precomputed_combined_dfs and os.path.exists(precomputed_combined_dfs[param_key]):
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-computation {i+1}/{len(all_parameter_combinations)}: Skipping NF={nf}, RS={rs}, GW={gw} (already processed and file exists).")
            continue 

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-computation {i+1}/{len(all_parameter_combinations)}: Loading models for NF={nf}, RS={rs}, GW={gw}")
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                loaded_rec_data = _load_recommendation_data(
                    genre_weight=gw,
                    num_features=nf,
                    new_user_regularization_strength=rs
                )
        
        if loaded_rec_data[0] is None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-computation {i+1}: Failed to load recommendation data. Skipping this config.")
            precomputed_combined_dfs[param_key] = "" # Store empty string to mark as failed/skipped
            save_checkpoint(precomputed_combined_dfs, PRECOMPUTED_CHECKPOINT_FILENAME) # Save progress even on failure
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-computation {i+1}: Collecting combined scores for all users...")
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                combined_dfs_for_this_config = collect_combined_dfs_for_single_run(
                    training_data_df=training_df,
                    loaded_data_tuple=loaded_rec_data,
                    new_user_regularization_strength=rs
                )
        
        if not combined_dfs_for_this_config.empty:
            # Save DataFrame to a temporary file and store its path
            filepath = save_intermediate_df_to_file(combined_dfs_for_this_config, param_key)
            if filepath: # Only update checkpoint if file was saved successfully
                precomputed_combined_dfs[param_key] = filepath
                save_checkpoint(precomputed_combined_dfs, PRECOMPUTED_CHECKPOINT_FILENAME) # Save after successful pre-computation of one config
            else:
                precomputed_combined_dfs[param_key] = "" # Mark as failed to save
                save_checkpoint(precomputed_combined_dfs, PRECOMPUTED_CHECKPOINT_FILENAME)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-computation {i+1}: No combined scores collected. Skipping.")
            precomputed_combined_dfs[param_key] = "" # Store empty string if no data
            save_checkpoint(precomputed_combined_dfs, PRECOMPUTED_CHECKPOINT_FILENAME) # Save progress even if empty result

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-computation complete.")


    # --- Loop through each defined subset for bagging evaluation ---
    total_subsets = len(subset_definitions)
    subset_progress = load_checkpoint(SUBSET_PROGRESS_CHECKPOINT_FILENAME) # Load subset progress
    
    current_subset_num = 0
    for subset_name in tqdm(
        sorted(subset_definitions.keys()),
        total=total_subsets,
        desc="Processing Subsets",
        unit="subset",
        leave=True # Keep this bar on screen as it's the top-level progress
    ):
        current_subset_num += 1
        current_parameter_combinations = subset_definitions[subset_name]
        num_bagging_runs_for_subset = len(current_parameter_combinations)
        
        # Determine last completed step for this subset
        last_completed_step = subset_progress.get(subset_name, STEP_INIT)

        if last_completed_step == STEP_EVALUATION_COMPLETE:
            print(f"\n--- Skipping subset '{subset_name}' ({current_subset_num}/{total_subsets}) as it is already complete. ---")
            continue
        
        if num_bagging_runs_for_subset == 0:
            print(f"\n--- Skipping subset '{subset_name}' ({current_subset_num}/{total_subsets}) as it contains no parameter combinations. ---")
            continue

        # Initialize DataFrames that might be loaded or computed.
        averaged_scores_df = None
        final_user_recs_df = None
        all_runs_combined_dfs_for_subset = [] # This list will be populated with DFs or remain empty if not needed

        print(f"\n--- Starting bagging for subset {current_subset_num}/{total_subsets}: '{subset_name}' with {num_bagging_runs_for_subset} runs ---")


        # --- NEW FLOW LOGIC: Try to load final_user_recs_df directly if we're past STEP_FINAL_RECS_SAVED ---
        # This is the primary optimization to skip re-computation
        force_recomputation = False # Flag to force re-computation if loading fails

        if last_completed_step >= STEP_FINAL_RECS_SAVED:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Final recommendations DataFrame already saved (from checkpoint). Attempting to load...")
            filepath_final_recs = subset_progress.get(f'{subset_name}_final_recs_path')
            
            if filepath_final_recs and os.path.exists(filepath_final_recs):
                final_user_recs_df = load_final_recs_df_from_file(filepath_final_recs)
                if final_user_recs_df is None or final_user_recs_df.empty:
                    # If loading fails (e.g., empty DF or corrupt), fall back to re-computation
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Failed to load saved final recommendations DataFrame or it was empty. Forcing full re-generation.")
                    force_recomputation = True
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Final recommendations DataFrame loaded successfully from disk.")
            else:
                # If path is missing or file doesn't exist, fall back to re-computation
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Path to saved final recommendations not found or file missing. Forcing full re-generation.")
                force_recomputation = True

        # If a fallback to re-computation is needed from the loading step, reset last_completed_step
        if force_recomputation:
            last_completed_step = STEP_INIT
            # Clear the old final recs path from checkpoint if we are forcing re-computation
            if f'{subset_name}_final_recs_path' in subset_progress:
                del subset_progress[f'{subset_name}_final_recs_path']
            save_checkpoint(subset_progress, SUBSET_PROGRESS_CHECKPOINT_FILENAME)


        # --- Main computation path if final_user_recs_df was NOT loaded ---
        # This block executes if final_user_recs_df is still None, meaning it needs to be computed.
        if final_user_recs_df is None: 
            # --- Step 1 & 2: Retrieve pre-computed combined_dfs for current subset and collect raw scores ---
            # These are needed to compute averaged_scores_df
            if last_completed_step < STEP_MODELS_LOADED_AND_COMBINED:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Collecting pre-computed data...")
                for i, (nf, rs, gw) in tqdm(
                    enumerate(current_parameter_combinations),
                    total=num_bagging_runs_for_subset,
                    desc=f"Loading models for '{subset_name}'",
                    unit="models",
                    leave=False
                ):
                    param_key = (nf, rs, gw)
                    filepath = precomputed_combined_dfs.get(param_key)
                    if filepath and os.path.exists(filepath):
                        loaded_df = load_intermediate_df_from_file(filepath)
                        if loaded_df is not None and not loaded_df.empty:
                            all_runs_combined_dfs_for_subset.append(loaded_df)
                        else:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}' | Run {i+1}: Loaded data for NF={nf}, RS={rs}, GW={gw} was empty or corrupted. Skipping.")
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}' | Run {i+1}: Skipping as pre-computed data for NF={nf}, RS={rs}, GW={gw} was not available or its file is missing/corrupted.")
                if not all_runs_combined_dfs_for_subset:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': No valid pre-computed data collected. Skipping evaluation for this subset.")
                    continue
                subset_progress[subset_name] = STEP_MODELS_LOADED_AND_COMBINED
                save_checkpoint(subset_progress, SUBSET_PROGRESS_CHECKPOINT_FILENAME)
            else:
                # If resuming and raw_dfs are needed (because final_user_recs_df was None), re-load them.
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Re-collecting pre-computed data (from checkpoint) for further steps...")
                for nf, rs, gw in current_parameter_combinations:
                    param_key = (nf, rs, gw)
                    filepath = precomputed_combined_dfs.get(param_key)
                    if filepath and os.path.exists(filepath):
                        loaded_df = load_intermediate_df_from_file(filepath)
                        if loaded_df is not None and not loaded_df.empty:
                            all_runs_combined_dfs_for_subset.append(loaded_df)
                if not all_runs_combined_dfs_for_subset:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Checkpoint loaded, but no valid temporary files found for re-collection. Skipping evaluation for this subset.")
                    continue


            # --- Step 3: Averaging scores ---
            if last_completed_step < STEP_SCORES_AVERAGED:
                print(f"\n--- Subset '{subset_name}': Averaging scores across {len(all_runs_combined_dfs_for_subset)} successful bagging runs ---")
                master_combined_df = pd.concat(all_runs_combined_dfs_for_subset, ignore_index=True)
                averaged_scores_df = master_combined_df.groupby(['User_id', 'Book Identifier'], as_index=False).agg(
                    CB_Weighted_Similarity_Score=('CB_Weighted_Similarity_Score', 'mean'),
                    CF_Predicted_Rating=('CF_Predicted_Rating', 'mean'),
                    Rated_by_User=('Rated by User', lambda x: x.any()),
                    Original_Rating=('Original Rating', 'first')
                )
                averaged_scores_df['CB_Weighted_Similarity_Score'] = averaged_scores_df['CB_Weighted_Similarity_Score'].fillna(0)
                averaged_scores_df['CF_Predicted_Rating'] = averaged_scores_df['CF_Predicted_Rating'].fillna(0)
                averaged_scores_df['Rated_by_User'] = averaged_scores_df['Rated_by_User'].astype(bool)

                subset_progress[subset_name] = STEP_SCORES_AVERAGED
                save_checkpoint(subset_progress, SUBSET_PROGRESS_CHECKPOINT_FILENAME)
            else: # last_completed_step is STEP_SCORES_AVERAGED or STEP_FINAL_RECS_GENERATED
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Re-concatenating models for averaging (from checkpoint)...")
                master_combined_df = pd.concat(all_runs_combined_dfs_for_subset, ignore_index=True)
                averaged_scores_df = master_combined_df.groupby(['User_id', 'Book Identifier'], as_index=False).agg(
                    CB_Weighted_Similarity_Score=('CB_Weighted_Similarity_Score', 'mean'),
                    CF_Predicted_Rating=('CF_Predicted_Rating', 'mean'),
                    Rated_by_User=('Rated by User', lambda x: x.any()),
                    Original_Rating=('Original Rating', 'first')
                )
                averaged_scores_df['CB_Weighted_Similarity_Score'] = averaged_scores_df['CB_Weighted_Similarity_Score'].fillna(0)
                averaged_scores_df['CF_Predicted_Rating'] = averaged_scores_df['CF_Predicted_Rating'].fillna(0)
                averaged_scores_df['Rated_by_User'] = averaged_scores_df['Rated_by_User'].astype(bool)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Scores already averaged (from checkpoint).")


            # --- Step 4 & Saving: Generate final recommendations and save ---
            # This block now always generates and saves if it's entered (i.e., final_user_recs_df wasn't loaded)
            if last_completed_step < STEP_FINAL_RECS_GENERATED: # Always true if we couldn't load final_user_recs_df initially
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Generating final recommendations using averaged scores...")
            else: # If we're here, it means last_completed_step was STEP_FINAL_RECS_GENERATED (2)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Re-generating final recommendations as it was not saved previously...")


            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    # averaged_scores_df is guaranteed to be available here, either freshly computed or re-derived from checkpoint.
                    final_user_recs_df = generate_recommendations_from_averaged_scores(
                        averaged_combined_df=averaged_scores_df,
                        output_limit_per_strategy=output_limit_per_strategy
                    )
            
            subset_progress[subset_name] = STEP_FINAL_RECS_GENERATED # Mark as generated
            save_checkpoint(subset_progress, SUBSET_PROGRESS_CHECKPOINT_FILENAME)

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Saving generated final recommendations DataFrame...")
            filepath_final_recs = save_final_recs_df_to_file(final_user_recs_df, subset_name)
            if filepath_final_recs:
                subset_progress[subset_name] = STEP_FINAL_RECS_SAVED # Mark as saved
                subset_progress[f'{subset_name}_final_recs_path'] = filepath_final_recs
                save_checkpoint(subset_progress, SUBSET_PROGRESS_CHECKPOINT_FILENAME)
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Failed to save final recommendations DataFrame. This might lead to issues on next run.")
        # Else for this section: final_user_recs_df was successfully loaded from disk already (handled at the very top).


        # --- Step 5: Evaluate the final averaged recommendations for this subset ---
        # Note: 'final_user_recs_df' is now guaranteed to be available either through generation or loading.
        if last_completed_step < STEP_EVALUATION_COMPLETE:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Evaluating final recommendations against relevant validation books...")
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    final_overlap_evaluation_df = evaluate_recommendations_overlap(
                        user_recommendations_df=final_user_recs_df,
                        relevant_validation_books_df=relevant_val_books_df
                    )
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Final evaluation complete.")

            # Display and save results for this subset
            output_dir = "bagging_evaluation_results"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_overlap_output_path = os.path.join(output_dir, f"bagging_overlap_evaluation_{subset_name}_{timestamp}.csv")
            
            print(f"\n--- Final Bagging Evaluation Results for Subset: '{subset_name}' ---")
            print(final_overlap_evaluation_df.to_string())

            try:
                final_overlap_evaluation_df.to_csv(final_overlap_output_path, index=False)
                print(f"\nFinal bagging evaluation results for subset '{subset_name}' saved to: {final_overlap_output_path}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Error saving final bagging evaluation results: {e}")
            
            # If successful, update checkpoint to complete
            subset_progress[subset_name] = STEP_EVALUATION_COMPLETE
            save_checkpoint(subset_progress, SUBSET_PROGRESS_CHECKPOINT_FILENAME)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Subset '{subset_name}': Evaluation already complete (from checkpoint).")
            # If the evaluation was already complete, we don't need to re-run it
            # nor re-save the output file, just indicate it was skipped.

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All subset bagging evaluations completed.")
