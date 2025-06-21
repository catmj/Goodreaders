import pandas as pd
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime # Import datetime for timestamps

# Explicitly import functions from your combined_recs_testing.py file
from combined_recs_testing_v1 import (
    _load_recommendation_data,
    _process_user_ratings_file,
    _split_user_data_for_testing,
    _get_relevant_books_from_validation,
    generate_all_recommendations_for_users,
    evaluate_recommendations_overlap
)


# --- Configuration for this Specific Test Run ---
NUM_FEATURES_FOR_TEST = 200
REGULARIZATION_STRENGTH_FOR_TEST = 10
GENRE_WEIGHT_FOR_TEST = 1


# 1. Define the path to your CSV file
csv_file_path = "../train_test_split/split_data_combined/cross_val_ratings.csv"


print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting recommendation pipeline...")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Configuration: Num Features={NUM_FEATURES_FOR_TEST}, Reg Strength={REGULARIZATION_STRENGTH_FOR_TEST}, Genre Weight={GENRE_WEIGHT_FOR_TEST}")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempting to process user ratings from: {csv_file_path}")


# Redirect stdout to suppress print statements from imported functions
with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        # 2. Load necessary recommendation data and models using the defined test parameters
        loaded_rec_data = _load_recommendation_data(
            genre_weight=GENRE_WEIGHT_FOR_TEST,
            num_features=NUM_FEATURES_FOR_TEST,
            new_user_regularization_strength=REGULARIZATION_STRENGTH_FOR_TEST
        )

# After the context manager, print statements resume normal behavior
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished loading recommendation models.")

if loaded_rec_data[0] is None: # Check if loading failed for content-based data
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to load recommendation data. Exiting example usage.")
else:
    # Encapsulate remaining calls within another redirection block to suppress their internal prints
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            # 3. Process the raw user ratings file
            user_data_processed_df = _process_user_ratings_file(csv_file_path)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished processing raw user ratings.")

    if not user_data_processed_df.empty:
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                # 4. Split user data into training and validation sets
                training_df, validation_df = _split_user_data_for_testing(
                    user_data_processed_df, validation_split_ratio=0.5
                )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished splitting user data into training and validation sets.")


        if not training_df.empty and not validation_df.empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating all recommendations for users...")
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    # 5. Generate recommendations for each user in the training set
                    all_user_recs_df = generate_all_recommendations_for_users(
                        training_data_df=training_df,
                        loaded_data_tuple=loaded_rec_data,
                        new_user_regularization_strength=REGULARIZATION_STRENGTH_FOR_TEST,
                        output_limit_per_strategy=60
                    )
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished generating all recommendations.")

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracting relevant books from validation set...")
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    # 6. Get relevant books from the validation set
                    relevant_val_books_df = _get_relevant_books_from_validation(
                        validation_df, relevance_threshold=3.0
                    )
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished extracting relevant books.")

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating recommendation overlap...")
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    # 7. Evaluate the overlap between recommendations and relevant validation books
                    overlap_evaluation_df = evaluate_recommendations_overlap(
                        user_recommendations_df=all_user_recs_df,
                        relevant_validation_books_df=relevant_val_books_df
                    )
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished evaluating recommendation overlap.")

            # 9. Save the overlap evaluation DataFrame to a CSV file
            output_dir = "evaluation_results"
            os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

            # Construct the filename with parameters
            filename = (
                f"overlap_evaluation_nf{NUM_FEATURES_FOR_TEST}"
                f"_rs{REGULARIZATION_STRENGTH_FOR_TEST}"
                f"_gw{GENRE_WEIGHT_FOR_TEST}.csv"
            )
            output_path = os.path.join(output_dir, filename)

            try:
                overlap_evaluation_df.to_csv(output_path, index=False)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Overlap evaluation results saved to: {output_path}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error saving evaluation results to CSV: {e}")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training or Validation DataFrame is empty after splitting. Cannot proceed with recommendation generation and evaluation.")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed user data DataFrame is empty. Cannot proceed with splitting or recommendations.")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recommendation pipeline completed.")
