import os
import pandas as pd
from datasets import load_dataset

import tqdm
import random
import hashlib

AVAILABLE_LANGUAGES = ['python', 'javascript', 'java', 'go', 'ruby', 'php']

def get_rosetta():
    try:

        print("Loading the Rosetta dataset...")
        # Try loading the dataset from Hugging Face
        dataset = load_dataset("christopher/rosetta-code", split="train")

        # Convert to a Pandas DataFrame if necessary
        dataset_df = dataset.to_pandas()
        
        # Save to a Parquet file
        root_dir = os.path.dirname(os.path.realpath(__file__))
        rosetta_raw_path = os.path.join(root_dir, 'rosetta/rosetta_raw.parquet')
        dataset_df.to_parquet(rosetta_raw_path, compression='brotli')
        print("Dataset loaded and saved as 'rosetta_raw.parquet'.")

    except Exception as e:

        print(f"Failed to load dataset from Hugging Face: {e}")
        # Fallback to loading from a local file if it exists
        local_path = "christopher_rosetta_code.parquet"

        if os.path.exists(local_path):
            print("Loading dataset from local file...")
            dataset = pd.read_parquet(local_path)
            
        else:
            print(f"Local file '{local_path}' not found.")
            return pd.DataFrame()  # Return an empty DataFrame if no data source is found

    return dataset


def process_rosetta(dataset):
    
    # Process and filter data
    processed_data = []
    generated_codes = set()  # Set to ensure unique base_code values

    def generate_seed(task):
        # Use hashlib to create a consistent hash seed across different systems
        hash_object = hashlib.md5(task.encode())
        return int(hash_object.hexdigest(), 16) % (10**8)

    for snippet in tqdm.tqdm(dataset, desc="Processing Rosetta Code snippets", total=len(dataset)):
        task_name = snippet['task_name']
        language_name = snippet['language_name']
        task_url = snippet['task_url']
        task_desc = snippet['task_description']
        code = snippet['code']

        # Check if language is in the available languages set
        if language_name.lower() not in AVAILABLE_LANGUAGES:
            continue

        # A CSV cell can only save text up to 32760 characters. (decide how to handle later...)
        # for now we will skip them...
        if len(code) >= 32700:
            continue

        # Generate a unique base_code with a consistent random format
        random.seed(generate_seed(task_name))  # Ensures consistency across machines
        base_code = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
        
        # Ensure the base_code is unique
        while base_code in generated_codes:
            base_code = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
        
        generated_codes.add(base_code)

        # Append relevant data to the processed list
        processed_data.append({
            'base_code_id': base_code,
            'task': task_name,
            'language': language_name.lower(),
            'task_url': task_url,
            'task_description': task_desc,
            'code': code
        })
    
    # Create and return DataFrame
    df = pd.DataFrame(processed_data)

    df['language'] = pd.Categorical(df['language'], categories=AVAILABLE_LANGUAGES, ordered=True)
    df = df.sort_values(by=['task', 'language'])

    return df