import pandas as pd
import numpy as np
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import pickle
from tqdm import tqdm
import pyarrow.parquet as pq

# --- Import visualization libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper function for statistical calculations (newly added) ---
def _calculate_descriptive_stats(column_data):
    """
    Calculates a comprehensive set of descriptive statistics for a pandas Series.

    Args:
        column_data (pd.Series): The Series of numeric data to analyze.

    Returns:
        dict: A dictionary containing the calculated statistics.
              Returns an empty dictionary if the series is empty or non-numeric after coercion.
    """
    stats_dict = {}

    # Ensure the column is numeric. If not, try to convert or skip.
    if not pd.api.types.is_numeric_dtype(column_data):
        original_rows = len(column_data)
        column_data = pd.to_numeric(column_data, errors='coerce')
        column_data.dropna(inplace=True) # Drop NaNs resulting from coercion
        if len(column_data) < original_rows:
            tqdm.write(f"    Warning: Non-numeric values found and removed during stats calculation.")

        if column_data.empty:
            return {} # Return empty if no valid numeric data

    if not column_data.empty:
        # Basic statistics
        mean_val = column_data.mean()
        median_val = column_data.median()
        mode_val = column_data.mode()
        skew_val = column_data.skew()
        range_val = column_data.max() - column_data.min()
        std_dev_val = column_data.std()

        # Additional metrics
        q1_val = column_data.quantile(0.25)
        q3_val = column_data.quantile(0.75)
        iqr_val = q3_val - q1_val

        percentile_10 = column_data.quantile(0.10)
        percentile_90 = column_data.quantile(0.90)
        percentile_95 = column_data.quantile(0.95)

        zero_count = (column_data == 0).sum()
        total_count = len(column_data)
        zero_proportion = (zero_count / total_count) if total_count > 0 else 0

        # Coefficient of Variation (handle division by zero if mean is zero)
        cv_val = (std_dev_val / mean_val) if mean_val != 0 else np.nan

        kurtosis_val = column_data.kurtosis()

        stats_dict = {
            'mean': mean_val,
            'median': median_val,
            'mode': mode_val.tolist() if not mode_val.empty else [],
            'skew': skew_val,
            'range': range_val,
            'std_dev': std_dev_val,
            'iqr': iqr_val,
            'percentile_10': percentile_10,
            'percentile_90': percentile_90,
            'percentile_95': percentile_95,
            'zero_count': zero_count,
            'zero_proportion': zero_proportion,
            'cv': cv_val,
            'kurtosis': kurtosis_val
        }
    return stats_dict


def _construct_file_path(checkpoint_root_dir, params):
    """
    Constructs the full file path for a given set of parameters.

    Args:
        checkpoint_root_dir (str): The root directory.
        params (dict): Dictionary with 'num_features', 'lambda_val', 'genre_weight'.

    Returns:
        str: The full path to the CSV file.
    """
    nf = params['num_features']
    rs = params['lambda_val']
    gw = params['genre_weight']
    config_output_dir = os.path.join(checkpoint_root_dir, f"nf{nf}_rs{rs}_gw{gw}")
    filename = f"overlap_evaluation_nf{nf}_rs{rs}_gw{gw}.csv"
    return os.path.join(config_output_dir, filename)

def _load_and_validate_dataframe(file_path):
    """
    Loads a CSV file into a DataFrame and performs initial validation.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame if valid, otherwise None.
    """
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}. Skipping.\n")
        return None

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"  File is empty: {file_path}. Skipping.\n")
            return None
        return df
    except pd.errors.EmptyDataError:
        print(f"  Error: The file {file_path} is empty or malformed. Skipping.\n")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred while loading {file_path}: {e}\n")
        return None

def _process_column_statistics(df):
    """
    Calculates various statistics for numeric columns in a DataFrame,
    excluding 'User_id' and 'Total_Relevant_Books'.
    Uses the _calculate_descriptive_stats helper function.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary where keys are column names and values are dictionaries
              containing the calculated statistics.
    """
    column_stats = {}
    # Exclude 'User_id' and 'Total_Relevant_Books' from overall analysis
    columns_to_analyze = [col for col in df.columns if col not in ['User_id', 'Total_Relevant_Books']]

    if not columns_to_analyze:
        return column_stats # Return empty if no columns to analyze

    for column in columns_to_analyze:
        stats = _calculate_descriptive_stats(df[column].copy())
        if stats: # Only add if statistics were successfully calculated
            column_stats[column] = stats
        else:
            print(f"    Skipping overall statistics for column '{column}' as it has no valid numeric data.")

    return column_stats

def _analyze_binned_overlap_data(df):
    """
    Analyzes overlap data by grouping users into bins based on 'Total_Relevant_Books'
    and calculating statistics for overlap columns within each bin.
    Uses the _calculate_descriptive_stats helper function.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze. Must contain 'Total_Relevant_Books'.

    Returns:
        dict: A dictionary where keys are bin labels (e.g., 'Low', 'Medium') and values
              are dictionaries containing overlap column statistics for that bin.
    """
    binned_results = {}
    
    # Define bins for Total_Relevant_Books
    bins = [0, 10, 50, 150, np.inf] # 0-10, 11-50, 51-150, 151+
    labels = ['Low (0-10)', 'Medium (11-50)', 'High (51-150)', 'Very High (151+)']

    # Ensure 'Total_Relevant_Books' is numeric and handle potential errors
    if not pd.api.types.is_numeric_dtype(df['Total_Relevant_Books']):
        df['Total_Relevant_Books'] = pd.to_numeric(df['Total_Relevant_Books'], errors='coerce')
        df.dropna(subset=['Total_Relevant_Books'], inplace=True)
        if df['Total_Relevant_Books'].empty:
            print("  Warning: 'Total_Relevant_Books' column became empty after numeric conversion. Skipping binned analysis.")
            return {} # Return empty if no data to bin

    # Create a 'book_bin' column
    df['book_bin'] = pd.cut(df['Total_Relevant_Books'], bins=bins, labels=labels, right=True, include_lowest=True)

    # Filter out rows where binning failed (e.g., NaN after conversion)
    df_binned = df.dropna(subset=['book_bin'])

    if df_binned.empty:
        print("  Warning: No data left after binning 'Total_Relevant_Books'. Skipping binned analysis.")
        return {}

    # Identify overlap columns (all columns starting with 'Overlap_Rec_')
    overlap_columns = [col for col in df_binned.columns if col.startswith('Overlap_Rec_')]

    if not overlap_columns:
        print("  Warning: No 'Overlap_Rec_' columns found for binned analysis.")
        return {}

    # Group by the created bins and calculate statistics for overlap columns
    for bin_label in labels:
        bin_df = df_binned[df_binned['book_bin'] == bin_label]

        if not bin_df.empty:
            bin_stats = {}
            for col in overlap_columns:
                stats = _calculate_descriptive_stats(bin_df[col].copy())
                if stats:
                    bin_stats[col] = stats
                else:
                    print(f"      Skipping stats for column '{col}' in bin '{bin_label}' as it has no valid numeric data.")
            if bin_stats: # Only add if there are actual stats for this bin
                binned_results[bin_label] = bin_stats
        else:
            print(f"  Warning: No data in bin '{bin_label}'. Skipping.")

    return binned_results


def _export_results(all_results, output_file_path=None):
    """
    Prints and/or exports the collected analysis results in a structured format,
    including overall and binned statistical metrics.

    Args:
        all_results (dict): A dictionary containing overall and binned analysis results for each file.
        output_file_path (str, optional): The path to the file where results should be written.
                                          If None, results are only printed to console.
    """
    output_lines = []

    header = "\n--- Analysis Results Summary ---"
    output_lines.append(header)
    print(header)

    if not all_results:
        no_results_msg = "No data files were successfully processed or no results were generated."
        output_lines.append(no_results_msg)
        print(no_results_msg)
    else:
        for file_path, file_data in all_results.items():
            file_header = f"\nResults for {file_path}:"
            output_lines.append(file_header)
            print(file_header)

            # Print Overall Statistics
            overall_stats = file_data.get('overall_stats', {})
            if overall_stats:
                output_lines.append("  --- Overall Statistics ---")
                print("  --- Overall Statistics ---")
                for column, stats_dict in overall_stats.items():
                    
                    if column != 'Total_Relevant_Books':
                        output_lines.append(f"  Column: {column}")
                        output_lines.append(f"    Mean: {stats_dict.get('mean', np.nan):.4f}")
                        output_lines.append(f"    Median: {stats_dict.get('median', np.nan):.4f}")
                        output_lines.append(f"    Mode(s): {', '.join(map(str, stats_dict.get('mode', []))) if stats_dict.get('mode') else 'No mode (or all values are unique/no data)'}")
                        output_lines.append(f"    Skew: {stats_dict.get('skew', np.nan):.4f}")
                        output_lines.append(f"    Range: {stats_dict.get('range', np.nan):.4f}")
                        output_lines.append(f"    Standard Deviation: {stats_dict.get('std_dev', np.nan):.4f}")
                        output_lines.append(f"    IQR: {stats_dict.get('iqr', np.nan):.4f}")
                        output_lines.append(f"    10th Percentile: {stats_dict.get('percentile_10', np.nan):.4f}")
                        output_lines.append(f"    90th Percentile: {stats_dict.get('percentile_90', np.nan):.4f}")
                        output_lines.append(f"    95th Percentile: {stats_dict.get('percentile_95', np.nan):.4f}")
                        output_lines.append(f"    Zero Count: {stats_dict.get('zero_count', 0)}")
                        output_lines.append(f"    Zero Proportion: {stats_dict.get('zero_proportion', np.nan):.4f}")
                        output_lines.append(f"    Coefficient of Variation (CV): {stats_dict.get('cv', np.nan):.4f}"
                                            if not np.isnan(stats_dict.get('cv', np.nan)) else "    Coefficient of Variation (CV): N/A (Mean is zero)")
                        output_lines.append(f"    Kurtosis: {stats_dict.get('kurtosis', np.nan):.4f}")
                        output_lines.append("-" * 30)

                        
                        for line in output_lines[-14:]:
                            print(line)
            else:
                output_lines.append("  No overall analyzable columns found for this file.")
                print("  No overall analyzable columns found for this file.")


            # Print Binned Statistics
            binned_stats = file_data.get('binned_stats', {})
            if binned_stats:
                output_lines.append("\n  --- Binned Statistics by Total_Relevant_Books ---")
                print("\n  --- Binned Statistics by Total_Relevant_Books ---")
                for bin_label, bin_data in binned_stats.items():
                    output_lines.append(f"  Bin: {bin_label}")
                    print(f"  Bin: {bin_label}")
                    for column, stats_dict in bin_data.items():
                        output_lines.append(f"    Column: {column}")
                        output_lines.append(f"      Mean: {stats_dict.get('mean', np.nan):.4f}")
                        output_lines.append(f"      Median: {stats_dict.get('median', np.nan):.4f}")
                        output_lines.append(f"      Mode(s): {', '.join(map(str, stats_dict.get('mode', []))) if stats_dict.get('mode') else '      Mode(s): No mode (or all values are unique/no data)'}")
                        output_lines.append(f"      Skew: {stats_dict.get('skew', np.nan):.4f}")
                        output_lines.append(f"      Range: {stats_dict.get('range', np.nan):.4f}")
                        output_lines.append(f"      Standard Deviation: {stats_dict.get('std_dev', np.nan):.4f}")
                        output_lines.append(f"      IQR: {stats_dict.get('iqr', np.nan):.4f}")
                        output_lines.append(f"      10th Percentile: {stats_dict.get('percentile_10', np.nan):.4f}")
                        output_lines.append(f"      90th Percentile: {stats_dict.get('percentile_90', np.nan):.4f}")
                        output_lines.append(f"      95th Percentile: {stats_dict.get('percentile_95', np.nan):.4f}")
                        output_lines.append(f"      Zero Count: {stats_dict.get('zero_count', 0)}")
                        output_lines.append(f"      Zero Proportion: {stats_dict.get('zero_proportion', np.nan):.4f}")
                        output_lines.append(f"      Coefficient of Variation (CV): {stats_dict.get('cv', np.nan):.4f}"
                                            if not np.isnan(stats_dict.get('cv', np.nan)) else "      Coefficient of Variation (CV): N/A (Mean is zero)")
                        output_lines.append(f"      Kurtosis: {stats_dict.get('kurtosis', np.nan):.4f}")
                        output_lines.append("-" * 30)

                        for line in output_lines[-14:]:
                            print(line)
            else:
                output_lines.append("  No binned analyzable data found for this file.")
                print("  No binned analyzable data found for this file.")

    if output_file_path:
        try:
            with open(output_file_path, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"\nAnalysis results successfully exported to: {output_file_path}")
        except Exception as e:
            print(f"\nError exporting results to {output_file_path}: {e}")


def _generate_and_save_plots(all_results, checkpoint_root_dir, parameters_list):
    """
    Generates and saves plots based on the analysis results.

    Args:
        all_results (dict): A dictionary containing overall and binned analysis results for each file.
        checkpoint_root_dir (str): The root directory where output directories are located.
        parameters_list (list of dict): The list of parameter sets used, to identify output folders.
    """
    print("\n--- Generating and Saving Plots ---")

    
    sns.set_theme(style="whitegrid")

    for params in parameters_list:
        file_path = _construct_file_path(checkpoint_root_dir, params)
        file_data = all_results.get(file_path)

        if file_data is None:
            print(f"  No analysis results found for {file_path}. Skipping plot generation.")
            continue

        nf = params['num_features']
        rs = params['lambda_val']
        gw = params['genre_weight']
        config_output_dir = os.path.join(checkpoint_root_dir, f"nf{nf}_rs{rs}_gw{gw}")
        plots_output_dir = os.path.join(config_output_dir, "plots")
        os.makedirs(plots_output_dir, exist_ok=True)
        
        print(f"  Generating plots for config: nf={nf}, rs={rs}, gw={gw}")

        # --- Plot 1: Overall Mean Overlap Bar Chart ---
        overall_stats = file_data.get('overall_stats', {})
        overlap_mean_data = {
            col.replace('Overlap_', ''): stats['mean']
            for col, stats in overall_stats.items()
            if col.startswith('Overlap_Rec_') and 'mean' in stats
        }

        if overlap_mean_data:
            df_overall_mean = pd.DataFrame(list(overlap_mean_data.items()), columns=['Strategy', 'Mean_Overlap'])
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Strategy', y='Mean_Overlap', data=df_overall_mean, palette='viridis')
            plt.title(f'Overall Mean Overlap (NF={nf}, RS={rs}, GW={gw})')
            plt.xlabel('Recommendation Strategy')
            plt.ylabel('Mean Number of Overlapping Books')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_filename = os.path.join(plots_output_dir, f'overall_mean_overlap_nf{nf}_rs{rs}_gw{gw}.png')
            plt.savefig(plot_filename)
            plt.close() 
            print(f"    Saved: {plot_filename}")
        else:
            print(f"    No overall mean overlap data to plot for config: nf={nf}, rs={rs}, gw={gw}")


        # --- Plot 2: Binned Mean Overlap Grouped Bar Chart ---
        binned_stats = file_data.get('binned_stats', {})
        if binned_stats:
            # Flatten binned_stats into a DataFrame for easier plotting
            binned_plot_data = []
            for bin_label, bin_data in binned_stats.items():
                for col, stats in bin_data.items():
                    if col.startswith('Overlap_Rec_') and 'mean' in stats:
                        binned_plot_data.append({
                            'Bin': bin_label,
                            'Strategy': col.replace('Overlap_', ''),
                            'Mean_Overlap': stats['mean']
                        })
            
            if binned_plot_data:
                df_binned_mean = pd.DataFrame(binned_plot_data)

                plt.figure(figsize=(12, 7))
                sns.barplot(x='Bin', y='Mean_Overlap', hue='Strategy', data=df_binned_mean, palette='magma')
                plt.title(f'Mean Overlap by Total Relevant Books Bin (NF={nf}, RS={rs}, GW={gw})')
                plt.xlabel('Total Relevant Books Bin')
                plt.ylabel('Mean Number of Overlapping Books')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Recommendation Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plot_filename = os.path.join(plots_output_dir, f'binned_mean_overlap_nf{nf}_rs{rs}_gw{gw}.png')
                plt.savefig(plot_filename)
                plt.close()
                print(f"    Saved: {plot_filename}")
            else:
                print(f"    No binned mean overlap data to plot for config: nf={nf}, rs={rs}, gw={gw}")
        else:
            print(f"    No binned statistics data found for config: nf={nf}, rs={rs}, gw={gw}")

    print("\n--- Plot Generation Complete ---")

def _plot_overlap_distributions(df, config_output_dir, params):
    """
    Generates and saves KDE plots for the distribution of overlap counts for each strategy.

    Args:
        df (pd.DataFrame): The DataFrame containing the overlap evaluation results.
        config_output_dir (str): The directory specific to the current configuration.
        params (dict): Dictionary with 'num_features', 'lambda_val', 'genre_weight'.
    """
    print("  Generating overlap distribution plots...")
    
    plots_output_dir = os.path.join(config_output_dir, "plots")
    os.makedirs(plots_output_dir, exist_ok=True)

    nf = params['num_features']
    rs = params['lambda_val']
    gw = params['genre_weight']

    overlap_columns = [col for col in df.columns if col.startswith('Overlap_Rec_')]

    if not overlap_columns:
        print("    No 'Overlap_Rec_' columns found for distribution plots.")
        return

    max_overlap = df[overlap_columns].max().max() if not df[overlap_columns].empty else 60
    x_limit = max(10, int(max_overlap * 1.1))


    for col in overlap_columns:
        plt.figure(figsize=(8, 5))
        
        sns.histplot(df[col], kde=True, bins=range(int(max_overlap) + 2), stat='density', color='skyblue', edgecolor='black', alpha=0.7)
     
        
        plt.title(f'Distribution of {col.replace("Overlap_", "")} Overlap (NF={nf}, RS={rs}, GW={gw})')
        plt.xlabel('Number of Overlapping Books')
        plt.ylabel('Density')
        plt.xlim(0, x_limit) # Set consistent x-axis limit
        plt.tight_layout()
        plot_filename = os.path.join(plots_output_dir, f'distribution_{col.lower()}_nf{nf}_rs{rs}_gw{gw}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"    Saved: {plot_filename}")

    print("  Overlap distribution plots generated.")


def analyze_overlap_data(checkpoint_root_dir, parameters_list, output_file_path=None):
    """
    Analyzes overlap data files by calculating various statistics for specified columns
    and generates plots. This is the main orchestration function.

    Args:
        checkpoint_root_dir (str): The root directory where the data files are located.
        parameters_list (list of dict): A list of dictionaries, each containing
                                        'num_features', 'lambda_val', and 'genre_weight'
                                        to construct file paths.
        output_file_path (str, optional): Path to a file where results will be saved.
                                          If None, results are only printed to console.
    """
    all_results = {}
    print(f"Starting data analysis in root directory: {checkpoint_root_dir}\n")

    for params in parameters_list:
        file_path = _construct_file_path(checkpoint_root_dir, params)
        print(f"Processing file: {file_path}")

        df = _load_and_validate_dataframe(file_path)
        if df is None:
            continue # Skip to next file if loading or validation failed

        # Perform overall column statistics (excluding Total_Relevant_Books)
        overall_stats = _process_column_statistics(df.copy()) # Pass a copy to avoid modifying original df for binning
        
        # Perform binned statistics (using Total_Relevant_Books)
        binned_stats = _analyze_binned_overlap_data(df.copy()) # Pass a copy for independent binning

        all_results[file_path] = {
            'overall_stats': overall_stats,
            'binned_stats': binned_stats
        }

        # Call the new plotting functions here for the current dataframe
        _generate_and_save_plots(all_results, checkpoint_root_dir, [params]) # Pass current params in a list
        _plot_overlap_distributions(df, os.path.join(checkpoint_root_dir, f"nf{params['num_features']}_rs{params['lambda_val']}_gw{params['genre_weight']}"), params)


    _export_results(all_results, output_file_path)


# --- Configuration ---
# where nfXXX_rsYYY_gwZZZ directories are located.
CHECKPOINT_ROOT_DIR = "single_run_evaluation_results_loop"

parameters_to_process = [
    {'num_features': 600, 'lambda_val': 5, 'genre_weight': 0.8}
]

# --- Output File Configuration ---
# Specify the path for the output file.
# If you leave this as None, results will only be printed to the console.
OUTPUT_RESULTS_FILE = "overlap_analysis_results.txt"


# --- Run the analysis ---
# Call the function with your root directory and parameters, and the output file path
analyze_overlap_data(CHECKPOINT_ROOT_DIR, parameters_to_process, OUTPUT_RESULTS_FILE)
