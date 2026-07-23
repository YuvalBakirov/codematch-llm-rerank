import os
import pandas as pd
import csv

import time
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv

from handle_rosetta import *
from cost_calculation import calculate_clones_creation, estimate_tokens_by_model
from handle_domain_subdomain import OpenAI_domain_subdomain_creation, add_domains_to_rosetta
from handle_clones import generate_clones

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROSETTA_DATA = os.path.join(ROOT_DIR, "rosetta_filtered.csv")
CLONE_PROMPTS = os.path.join(ROOT_DIR, "clone_prompts.csv")

ORIGINAL_CODE_PATH = os.path.join(ROOT_DIR, "evaluation-datasets/original_code_benchmark.csv")
TEST_CODE_PATH = os.path.join(ROOT_DIR, "evaluation-datasets/test_code_benchmark.csv")

MODEL_NAME = "gpt-4o-mini"

def write_to_csv(data, csv_file):
    """
    Writes the metadata and code into a CSV.
    """
        # Assuming 'original_code_df' is the DataFrame you created
    fieldnames = data.columns

    # Convert the fieldnames to a list (since DataFrame.columns returns an Index object)
    fieldnames = fieldnames.tolist()

    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as file:

        # Create a DictWriter object, passing in the fieldnames
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Check if the file is empty, and if so, write the header
        if file.tell() == 0:
            writer.writeheader()
        
        # Write the DataFrame to the CSV row by row
        writer.writerows(data.to_dict(orient='records'))



def process_batch(original_code_data, benchmark_test_data):
    """
    Processes a batch of original code and benchmark test data, writing to CSV.
    """
    write_to_csv(original_code_data, ORIGINAL_CODE_PATH)
    write_to_csv(benchmark_test_data, TEST_CODE_PATH)

    # Clear data after writing
    original_code_data = pd.DataFrame(columns=['base_code_id', 'language', 'task', 'domain', 'subdomain', 'code'])
    benchmark_test_data = pd.DataFrame(columns=['clone_code_id', 'base_code_id', 'task', 'domain', 'subdomain', 'clone_language', 'clone_type', 'clone_sub_type', 'code'])

    return original_code_data, benchmark_test_data



def create_evaluation_datasets(client, rosetta_data, batch_size=100, max_items=None):
    """
    Process Rosetta dataset and generate original code and code clones.
    """
    start_time = time.time()
    previous_batch_time = start_time

    batch_count = 1
    print(f"Initializing Batch {batch_count}:-----------------------------------")

    original_code_data = pd.DataFrame(columns=['base_code_id', 'language', 'task', 'domain', 'subdomain', 'code'])
    benchmark_test_data = pd.DataFrame(columns=['clone_code_id', 'base_code_id', 'task', 'domain', 'subdomain', 'clone_language', 'clone_type', 'clone_sub_type', 'code'])

    rosetta_python = rosetta_data[rosetta_data['language'] == 'python']

    # print(rosetta_python.head())

    clone_types = pd.read_csv(CLONE_PROMPTS)

    if max_items:
        number_of_iterations = min(max_items, len(rosetta_python))
    else:
        number_of_iterations = len(rosetta_python)

    for iteration_count, (idx, item) in enumerate(tqdm.tqdm(rosetta_python.iterrows(), total=number_of_iterations, desc="Processing Rosetta Data:"), start=1):
        # Stop if we've processed the specified number of items
        if max_items and iteration_count > max_items:
            break

        print(f" Task: {iteration_count}, token input size: {estimate_tokens_by_model(item['code'], MODEL_NAME)}")

        base_code_id = item['base_code_id']
        language = item['language']
        task = item['task']
        domain = item['domain'] if 'domain' in item else "Unknown"
        subdomain = item['subdomain']  if 'subdomain' in item else "Unknown"
        source_code = item["code"]

        # Add to original code dataset
        original_code_data = original_code_data.append({
            'base_code_id': base_code_id,
            'language': language,
            'task': task,
            'domain': domain,
            'subdomain': subdomain,
            'code': source_code
        }, ignore_index=True)

        # Create clones
        benchmark_test_data = generate_clones(client, source_code, base_code_id, domain, subdomain, task, clone_types, rosetta_data, benchmark_test_data)
       
        # Process in batches to avoid memory issues
        if (iteration_count + 1) % batch_size == 0:
            batch_time = (time.time() - previous_batch_time) / 60
            print(f"Batch {batch_count} - Time taken: {batch_time:.2f} minutes\n")

            original_code_data, benchmark_test_data = process_batch(original_code_data, benchmark_test_data)

            batch_count += 1
            previous_batch_time = time.time()
            print(f"Initializing Batch {batch_count}:-----------------------------------")

    # Process any remaining data that didn't complete a full batch
    if not original_code_data.empty or not benchmark_test_data.empty:
        process_batch(original_code_data, benchmark_test_data)

    # Total time calculation
    total_time = (time.time() - start_time) / 60
    print(f"Total time taken: {total_time:.2f} minutes")





def handle_evaluation_datasets(client, rosetta_with_domains_subdomains):

    while True:
        response = input("\nDo you want to proceed? (y/n): ").strip().lower()
        if response == 'y':
            print("Proceeding...\n")
            break
        elif response == 'n':
            print("Stopping...")
            exit()  # Stop the program
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


    # Evaluation datasets creation
    # root_dir = os.path.dirname(os.path.realpath(__file__))
    # original_code_path = os.path.join(root_dir, 'evaluation-datasets/original_code_benchmark.csv')
    # test_code_path = os.path.join(root_dir, 'evaluation-datasets/test_code_benchmark.csv')
    

    if os.path.exists(ORIGINAL_CODE_PATH) and os.path.exists(TEST_CODE_PATH):
        print("There are already original code and test code csvs.\n")
        original_code = pd.read_csv(ORIGINAL_CODE_PATH)
        test_code = pd.read_csv(TEST_CODE_PATH)

        while True:
            response = input("\nDo you want to run anyways? (y/n): ").strip().lower()
            if response == 'y':
                print("Proceeding...\n")
                break
            elif response == 'n':
                print("Stopping...")
                exit()  # Stop the program
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        if response == 'y':
            create_evaluation_datasets(client, rosetta_with_domains_subdomains, batch_size=5)
    else:
        print("No existing csv in folder. Proceeding with creating one...")
        create_evaluation_datasets(client, rosetta_with_domains_subdomains, batch_size=5)


 

def main():

    rosetta = get_rosetta()
    rosetta = process_rosetta(rosetta)

    if rosetta.empty:
        print("No data processed.")

    unique_task_df = rosetta[rosetta['language'] == 'python']
    
    load_dotenv()
    OPENAI_KEY = os.environ.get('OPENAI_KEY')
    client = OpenAI(api_key=OPENAI_KEY)

    unique_task_with_domains = OpenAI_domain_subdomain_creation(client, unique_task_df)

    rosetta_with_domains_subdomains = add_domains_to_rosetta(rosetta, unique_task_with_domains)

    calculate_clones_creation(unique_task_df)
    handle_evaluation_datasets(client, rosetta_with_domains_subdomains)


if __name__ == "__main__":
    main()