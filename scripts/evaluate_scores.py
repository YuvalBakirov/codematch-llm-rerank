import os
import pandas as pd
from datetime import datetime
from core.metrics import *  # Importing necessary functions from metrics
import config  # Importing configuration module


def resolve_model_path(model_name, folder_type):
    """
    Generate the absolute path for the specified model and folder type.

    Args:
        model_name (str): Name of the model.
        folder_type (str): Type of folder ('global_clone' or 'direct_clone').

    Returns:
        str: Absolute path to the specified folder.
    """
    model_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), f"output/{model_name}"))
    return os.path.join(model_folder, folder_type)


def get_latest_csv(folder, prefix):
    """
    Retrieve the most recent CSV file in a folder that matches a specific prefix 
    and contains a valid timestamp in its filename.

    Args:
        folder (str): Path to the folder containing the files.
        prefix (str): Prefix to filter files (e.g., "codebert-base_global_clone_search_scores").

    Returns:
        str: Path to the latest CSV file.

    Raises:
        ValueError: If the folder doesn't exist or no valid files are found.
    """
    # Ensure the specified folder exists
    if not os.path.exists(folder):
        raise ValueError(f"Folder does not exist: {folder}")
    
    # Find files that match the prefix and end with '.csv'
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".csv")]
    
    # Filter files with valid timestamps
    valid_files = []
    for f in files:
        try:
            timestamp_part = f[len(prefix) + 1:].replace(".csv", "").strip()
            datetime.strptime(timestamp_part, "%d.%m.%Y_%H-%M-%S")
            valid_files.append(f)
        except ValueError:
            continue  # Skip files with invalid timestamps

    # Check if any valid files are found
    if not valid_files:
        raise ValueError(f"No valid files with prefix '{prefix}' found in folder '{folder}'")
    
    # Identify the latest file by comparing timestamps
    latest_file = max(valid_files, key=lambda f: datetime.strptime(f[len(prefix) + 1:].replace(".csv", "").strip(), "%d.%m.%Y_%H-%M-%S"))
    return os.path.join(folder, latest_file)


def get_scores_df(model_name, folder_type):
    """
    Load the most recent evaluation scores CSV for a given model and folder type.

    Args:
        model_name (str): Name of the model.
        folder_type (str): Type of folder ('global-clone' or 'direct-clone').

    Returns:
        pd.DataFrame: DataFrame containing the evaluation scores.

    Raises:
        None: Prints an error message if processing fails and returns None.
    """
    folder_path = resolve_model_path(model_name, folder_type)
    try:
        # Define the prefix for score files
        scores_prefix = f"{model_name}_global_clone_search_scores"

        # Locate the latest scores CSV file
        latest_scores_csv = get_latest_csv(folder_path, scores_prefix)
        
        # Load the scores into a DataFrame
        scores_df = pd.read_csv(latest_scores_csv)
        print(f"Successfully loaded the latest evaluation for {model_name} from {folder_type}: {latest_scores_csv}")

        return scores_df

    except ValueError as e:
        print(f"Error processing model '{model_name}' in folder '{folder_type}': {e}")
        return None


def main():
    """
    Main function to evaluate models using the specified configuration.
    Iterates through available models and evaluates their global clone search results.
    """

    global_search_logger = logging.getLogger("benchmark.global_search")

    for model_name in config.AVAILABLE_MODELS:
        # Load evaluation scores for the 'global-clone' folder type
        scores_df = get_scores_df(model_name, "global-clone")
        
        # Evaluate global search results if the DataFrame is valid
        if scores_df is not None:
            evaluate_global_search_results(global_search_logger, scores_df, model_name)


# Entry point for script execution
if __name__ == "__main__":
    main()
