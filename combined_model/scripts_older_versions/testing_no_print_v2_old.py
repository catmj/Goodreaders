import pandas as pd
import numpy as np
import os
import sys
from contextlib import redirect_stdout, redirect_stderr # Import redirect_stderr
from datetime import datetime
import pickle # Import pickle for saving/loading checkpoints
from tqdm import tqdm # Import tqdm for progress bars
import pyarrow.parquet as pq # Import pyarrow for Parquet file handling

# Explicitly import functions from your combined_recs_testing_3.py file
# IMPORTANT: Please ensure 'combined_recs_testing_3.py' is the correct file on your system
# and is accessible from this script's location.
from combined_recs_testing_v3_old import (
    _load_recommendation_data,
    _process_user_ratings_file,
    _split_user_data_for_testing,
    _get_relevant_books_from_validation,
    generate_all_recommendations_for_users,
    evaluate_recommendations_overlap
)

# --- Configuration for the Overall Test Run ---
# This script will now loop through these configurations
test_configs = [
    {'num_features': 500, 'lambda_val': 15, 'genre_weight': 1},
    {'num_features': 600, 'lambda_val': 15, 'genre_weight': 1},
    {'num_features': 800, 'lambda_val': 15, 'genre_weight': 1},

    {'num_features': 400, 'lambda_val': 10, 'genre_weight': 1},
    {'num_features': 400, 'lambda_val': 5, 'genre_weight': 1},
    {'num_features': 400, 'lambda_val': 1, 'genre_weight': 1},

    {'num_features': 500, 'lambda_val': 10, 'genre_weight': 1},
    {'num_features': 600, 'lambda_val': 5, 'genre_weight': 1},
    {'num_features': 800, 'lambda_val': 10, 'genre_weight': 1},

    {'num_features': 500, 'lambda_val': 10, 'genre_weight': 1},
    {'num_features': 600, 'lambda_val': 5, 'genre_weight': 1},
    {'num_features': 800, 'lambda_val': 10, 'genre_weight': 1},

    {'num_features': 400, 'lambda_val': 8, 'genre_weight': 1},
    {'num_features': 500, 'lambda_val': 8, 'genre_weight': 1},
    {'num_features': 600, 'lambda_val': 8, 'genre_weight': 1},
    {'num_features': 600, 'lambda_val': 10, 'genre_weight': 1},

    {'num_features': 500, 'lambda_val': 10, 'genre_weight': 0.8},
    {'num_features': 500, 'lambda_val': 10, 'genre_weight': 0.5},
    {'num_features': 500, 'lambda_val': 10, 'genre_weight': 0},

    {'num_features': 600, 'lambda_val': 5, 'genre_weight': 0.8},
    {'num_features': 600, 'lambda_val': 5, 'genre_weight': 0.5},
    {'num_features': 600, 'lambda_val': 5, 'genre_weight': 0}

    
]


# --- Checkpointing and Output Paths ---
CHECKPOINT_ROOT_DIR = "single_run_evaluation_results_loop" # Changed root directory
# Checkpoint for overall progress across all configurations
OVERALL_PROGRESS_CHECKPOINT_FILENAME = os.path.join(CHECKPOINT_ROOT_DIR, "overall_pipeline_progress.pkl")

# Define step markers for checkpointing within each configuration's pipeline
STEP_INIT = -1
STEP_DATA_LOADED = 0
STEP_USER_DATA_PROCESSED = 1
STEP_DATA_SPLIT = 2
STEP_RECS_GENERATED = 3
STEP_RELEVANT_BOOKS_EXTRACTED = 4
STEP_EVALUATION_COMPLETE = 5

# --- Checkpointing Helper Functions ---
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

def save_df_to_parquet(df: pd.DataFrame, filepath: str):
    """Saves a DataFrame to a Parquet file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        df.to_parquet(filepath, index=False)
        return True
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving DataFrame to {filepath}: {e}")
        return False

def load_df_from_parquet(filepath: str) -> pd.DataFrame | None:
    """Loads a DataFrame from a Parquet file."""
    if os.path.exists(filepath):
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading DataFrame from {filepath}: {e}")
            return None
    return None

# 1. Define the path to your CSV file
csv_file_path = "../train_test_split/split_data_combined/cross_val_ratings.csv"


print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting recommendation pipeline loop...")

# Load overall pipeline progress
overall_pipeline_progress = load_checkpoint(OVERALL_PROGRESS_CHECKPOINT_FILENAME)

total_configs = len(test_configs)
for i, config in enumerate(tqdm(test_configs, desc="Overall Test Progress", unit="config", position=0)):
    nf = config['num_features']
    rs = config['lambda_val']
    gw = config['genre_weight']
    
    config_key = (nf, rs, gw) # Unique identifier for this configuration
    
    # Each config will have its own subdirectory for temporary dataframes
    config_output_dir = os.path.join(CHECKPOINT_ROOT_DIR, f"nf{nf}_rs{rs}_gw{gw}")
    os.makedirs(config_output_dir, exist_ok=True)

    # Load progress for the current configuration
    current_config_progress = overall_pipeline_progress.get(config_key, {})
    last_completed_step = current_config_progress.get('last_completed_step', STEP_INIT)

    if last_completed_step == STEP_EVALUATION_COMPLETE:
        print(f"\n--- Skipping Config {i+1}/{total_configs}: Features={nf}, Reg Strength={rs}, Genre Weight={gw} (already completed) ---")
        continue

    print(f"\n################################################################################")
    print(f"### Running Config {i+1}/{total_configs}: Features={nf}, Reg Strength={rs}, Genre Weight={gw} ###")
    print(f"################################################################################")
    
    # Initialize variables for the current run. They will be populated as steps complete or loaded.
    loaded_rec_data = None
    user_data_processed_df = pd.DataFrame()
    training_df = pd.DataFrame()
    validation_df = pd.DataFrame()
    all_user_recs_df = pd.DataFrame()
    relevant_val_books_df = pd.DataFrame()


    # --- Stage 1: Load Recommendation Data ---
    if last_completed_step < STEP_DATA_LOADED:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading recommendation models...")
        # Suppress all internal prints (stdout and stderr) from the imported functions
        with open(os.devnull, 'w') as fnull: 
            with redirect_stdout(fnull), redirect_stderr(fnull): # Redirect stderr as well
                loaded_rec_data = _load_recommendation_data(
                    genre_weight=gw,
                    num_features=nf,
                    new_user_regularization_strength=rs
                )
        if loaded_rec_data[0] is None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load recommendation data for this config. Skipping.")
            continue # Skip to next configuration
        current_config_progress['loaded_rec_data'] = loaded_rec_data # Save the loaded tuple
        current_config_progress['last_completed_step'] = STEP_DATA_LOADED
        overall_pipeline_progress[config_key] = current_config_progress
        save_checkpoint(overall_pipeline_progress, OVERALL_PROGRESS_CHECKPOINT_FILENAME)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished loading recommendation models.")
    else:
        loaded_rec_data = current_config_progress['loaded_rec_data']
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recommendation models already loaded (from checkpoint).")


    # --- Stage 2: Process User Ratings File ---
    if last_completed_step < STEP_USER_DATA_PROCESSED:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing raw user ratings...")
        with open(os.devnull, 'w') as fnull: # Suppress internal prints (stdout and stderr)
            with redirect_stdout(fnull), redirect_stderr(fnull):
                user_data_processed_df = _process_user_ratings_file(csv_file_path)
        if user_data_processed_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed user data DataFrame is empty. Exiting pipeline.")
            continue
        
        df_path = os.path.join(config_output_dir, 'user_data_processed.parquet')
        if not save_df_to_parquet(user_data_processed_df, df_path):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to save user_data_processed_df. Skipping config.")
            continue
        current_config_progress['user_data_processed_df_path'] = df_path
        current_config_progress['last_completed_step'] = STEP_USER_DATA_PROCESSED
        overall_pipeline_progress[config_key] = current_config_progress
        save_checkpoint(overall_pipeline_progress, OVERALL_PROGRESS_CHECKPOINT_FILENAME)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished processing raw user ratings.")
    else:
        user_data_processed_df = load_df_from_parquet(current_config_progress['user_data_processed_df_path'])
        if user_data_processed_df is None or user_data_processed_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load user_data_processed_df from checkpoint. Re-running this stage.")
            last_completed_step = STEP_INIT # Force re-run
            continue # Re-start loop iteration to re-process this stage
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User data already processed (from checkpoint).")


    # --- Stage 3: Split User Data ---
    if last_completed_step < STEP_DATA_SPLIT:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Splitting user data into training and validation sets...")
        # Progress bar for _split_user_data_for_testing is handled internally using tqdm.write().
        # So we keep redirecting stdout/stderr for other potential prints from helper functions.
        with open(os.devnull, 'w') as fnull: 
            with redirect_stdout(fnull), redirect_stderr(fnull): # Redirect stderr as well
                training_df, validation_df = _split_user_data_for_testing(
                    user_data_processed_df, validation_split_ratio=0.5
                )
        if training_df.empty or validation_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training or Validation DataFrame is empty after splitting. Skipping config.")
            continue
        
        training_df_path = os.path.join(config_output_dir, 'training_df.parquet')
        validation_df_path = os.path.join(config_output_dir, 'validation_df.parquet')
        if not save_df_to_parquet(training_df, training_df_path) or \
           not save_df_to_parquet(validation_df, validation_df_path):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to save split DataFrames. Skipping config.")
            continue
        current_config_progress['training_df_path'] = training_df_path
        current_config_progress['validation_df_path'] = validation_df_path
        current_config_progress['last_completed_step'] = STEP_DATA_SPLIT
        overall_pipeline_progress[config_key] = current_config_progress
        save_checkpoint(overall_pipeline_progress, OVERALL_PROGRESS_CHECKPOINT_FILENAME)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished splitting user data.")
    else:
        training_df = load_df_from_parquet(current_config_progress['training_df_path'])
        validation_df = load_df_from_parquet(current_config_progress['validation_df_path'])
        if training_df is None or training_df.empty or validation_df is None or validation_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load split DataFrames from checkpoint. Re-running this stage.")
            last_completed_step = STEP_INIT # Force re-run
            continue # Re-start loop iteration to re-process this stage
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User data already split (from checkpoint).")


    # --- Stage 4: Generate All Recommendations for Users ---
    if last_completed_step < STEP_RECS_GENERATED:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating all recommendations for users...")
        # Progress bar for generate_all_recommendations_for_users is handled internally using tqdm.write().
        # We allow tqdm's own progress bar to be visible, so no redirection for this main call,
        # but ensure nested calls still redirect if they were to use print directly.
        # This function should internally use tqdm.write() for all its prints.
        all_user_recs_df = generate_all_recommendations_for_users(
            training_data_df=training_df,
            loaded_data_tuple=loaded_rec_data,
            new_user_regularization_strength=rs, # Use dynamic rs
            output_limit_per_strategy=60
        )
        if all_user_recs_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No recommendations generated. Skipping config.")
            continue
        
        df_path = os.path.join(config_output_dir, 'all_user_recs.parquet')
        if not save_df_to_parquet(all_user_recs_df, df_path):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to save all_user_recs_df. Skipping config.")
            continue
        current_config_progress['all_user_recs_df_path'] = df_path
        current_config_progress['last_completed_step'] = STEP_RECS_GENERATED
        overall_pipeline_progress[config_key] = current_config_progress
        save_checkpoint(overall_pipeline_progress, OVERALL_PROGRESS_CHECKPOINT_FILENAME)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished generating all recommendations.")
    else:
        all_user_recs_df = load_df_from_parquet(current_config_progress['all_user_recs_df_path'])
        if all_user_recs_df is None or all_user_recs_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load all_user_recs_df from checkpoint. Re-running this stage.")
            last_completed_step = STEP_INIT # Force re-run
            continue # Re-start loop iteration to re-process this stage
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All recommendations already generated (from checkpoint).")


    # --- Stage 5: Extract Relevant Books from Validation Set ---
    if last_completed_step < STEP_RELEVANT_BOOKS_EXTRACTED:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracting relevant books from validation set...")
        # Progress bar for _get_relevant_books_from_validation is handled internally using tqdm.write().
        with open(os.devnull, 'w') as fnull: 
            with redirect_stdout(fnull), redirect_stderr(fnull): # Redirect stderr as well
                relevant_val_books_df = _get_relevant_books_from_validation(
                    validation_df, relevance_threshold=3.0
                )
        if relevant_val_books_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No relevant validation books found. Skipping config.")
            continue
        
        df_path = os.path.join(config_output_dir, 'relevant_val_books.parquet')
        if not save_df_to_parquet(relevant_val_books_df, df_path):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to save relevant_val_books_df. Skipping config.")
            continue
        current_config_progress['relevant_val_books_df_path'] = df_path
        current_config_progress['last_completed_step'] = STEP_RELEVANT_BOOKS_EXTRACTED
        overall_pipeline_progress[config_key] = current_config_progress
        save_checkpoint(overall_pipeline_progress, OVERALL_PROGRESS_CHECKPOINT_FILENAME)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished extracting relevant books.")
    else:
        relevant_val_books_df = load_df_from_parquet(current_config_progress['relevant_val_books_df_path'])
        if relevant_val_books_df is None or relevant_val_books_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load relevant_val_books_df from checkpoint. Re-running this stage.")
            last_completed_step = STEP_INIT # Force re-run
            continue # Re-start loop iteration to re-process this stage
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Relevant books already extracted (from checkpoint).")


    # --- Stage 6: Evaluate Recommendation Overlap ---
    if last_completed_step < STEP_EVALUATION_COMPLETE:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating recommendation overlap...")
        # Progress bar for evaluate_recommendations_overlap is handled internally using tqdm.write().
        # We allow tqdm's own progress bar to be visible, so no redirection for this main call.
        overlap_evaluation_df = evaluate_recommendations_overlap(
            user_recommendations_df=all_user_recs_df,
            relevant_validation_books_df=relevant_val_books_df
        )
        if overlap_evaluation_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No overlap evaluation results. Skipping config.")
            continue
        
        # Save the overlap evaluation DataFrame to a CSV file
        # Construct the filename with parameters
        filename = (
            f"overlap_evaluation_nf{nf}"
            f"_rs{rs}"
            f"_gw{gw}.csv"
        )
        output_path = os.path.join(config_output_dir, filename) # Save to config-specific directory

        try:
            overlap_evaluation_df.to_csv(output_path, index=False)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Overlap evaluation results saved to: {output_path}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving evaluation results to CSV: {e}. Skipping checkpoint update.")
            continue # Do not mark as complete if save failed

        current_config_progress['last_completed_step'] = STEP_EVALUATION_COMPLETE
        overall_pipeline_progress[config_key] = current_config_progress
        save_checkpoint(overall_pipeline_progress, OVERALL_PROGRESS_CHECKPOINT_FILENAME)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished evaluating recommendation overlap.")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recommendation evaluation already complete (from checkpoint).")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recommendation pipeline loop completed.")
