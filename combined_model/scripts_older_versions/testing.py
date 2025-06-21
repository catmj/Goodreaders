import pandas as pd
import os

# Explicitly import functions and constants from your combined_recs_testing.py file
from combined_recs_testing_v1 import (
    _load_recommendation_data,
    _process_user_ratings_file,
    _split_user_data_for_testing,
    _get_relevant_books_from_validation,
    generate_all_recommendations_for_users,
    evaluate_recommendations_overlap
)


# --- Configuration for this Specific Test Run ---
NUM_FEATURES_FOR_TEST = 400
REGULARIZATION_STRENGTH_FOR_TEST = 15
GENRE_WEIGHT_FOR_TEST = 1


# 1. Define the path to your CSV file
csv_file_path = "../train_test_split/split_data_combined/cross_val_ratings.csv"

print(f"Attempting to process user ratings from: {csv_file_path}")

# 2. Load necessary recommendation data and models using the defined test parameters
loaded_rec_data = _load_recommendation_data(
    genre_weight=GENRE_WEIGHT_FOR_TEST,
    num_features=NUM_FEATURES_FOR_TEST,
    new_user_regularization_strength=REGULARIZATION_STRENGTH_FOR_TEST
)

if loaded_rec_data[0] is None: # Check if loading failed for content-based data
    print("Failed to load recommendation data. Exiting example usage.")
else:
    # 3. Process the raw user ratings file
    user_data_processed_df = _process_user_ratings_file(csv_file_path)

    if not user_data_processed_df.empty:
        # 4. Split user data into training and validation sets
        training_df, validation_df = _split_user_data_for_testing(
            user_data_processed_df, validation_split_ratio=0.5
        )

        if not training_df.empty and not validation_df.empty:
            # 5. Generate recommendations for each user in the training set
            all_user_recs_df = generate_all_recommendations_for_users(
                training_data_df=training_df,
                loaded_data_tuple=loaded_rec_data,
                new_user_regularization_strength=REGULARIZATION_STRENGTH_FOR_TEST, # Use the test parameter
                output_limit_per_strategy=20 
            )

            # 6. Get relevant books from the validation set
            relevant_val_books_df = _get_relevant_books_from_validation(
                validation_df, relevance_threshold=3.0 # Default relevance threshold
            )

            # 7. Evaluate the overlap between recommendations and relevant validation books
            overlap_evaluation_df = evaluate_recommendations_overlap(
                user_recommendations_df=all_user_recs_df,
                relevant_validation_books_df=relevant_val_books_df
            )

            # 8. Display the evaluation results (e.g., first few rows)
            print("\n--- Recommendation Overlap Evaluation Results (first 5 rows) ---")
            print(overlap_evaluation_df.head())

            # Optional: Calculate average overlaps or other metrics
            if not overlap_evaluation_df.empty:
                print("\n--- Average Overlap Counts Across Users ---")
                # Exclude 'User_id' and 'Total_Relevant_Books' for averaging overlap columns
                overlap_cols = [col for col in overlap_evaluation_df.columns if col.startswith('Overlap_')]
                print(overlap_evaluation_df[overlap_cols].mean())
                print(f"\nAverage Total Relevant Books: {overlap_evaluation_df['Total_Relevant_Books'].mean():.2f}")

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
                    print(f"\nOverlap evaluation results saved to: {output_path}")
                except Exception as e:
                    print(f"Error saving evaluation results to CSV: {e}")

        else:
            print("Training or Validation DataFrame is empty after splitting. Cannot proceed with recommendation generation and evaluation.")
    else:
        print("Processed user data DataFrame is empty. Cannot proceed with splitting or recommendations.")
