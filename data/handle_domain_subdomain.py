import pandas as pd
import os
import csv

from cost_calculation import calculate_cost, domain_subdomain_calculation_func, identify_domains_and_subdomains

MODEL_NAME = "gpt-4o-mini"

def get_available_domains(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Group subdomains by domain
    grouped = df.groupby('domain')['subdomain'].apply(list).reset_index()

    # Create the formatted available_domains variable
    available_domains = "Available domains and subdomains are as follows:\n"
    for i, row in enumerate(grouped.itertuples(), 1):
        domain = row.domain
        subdomains = ', '.join(row.subdomain)
        available_domains += f"{i}. Domain: {domain}; Subdomains: {subdomains}\n"

    return available_domains


def OpenAI_domain_subdomain_creation(client, unique_task_df):

    root_dir = os.path.dirname(os.path.realpath(__file__))
    domains_and_subdomains_path = os.path.join(root_dir, "rosetta/domains-and-subdomains.csv")
    available_domains = get_available_domains(domains_and_subdomains_path)

    system_prompt = (
        "You are an expert software categorization assistant. "
        "Given a task, its description, and the code provided, analyze and determine the domain (field of work) "
        "and subdomain (specific subfield) the code best belongs to. "
        "Consider the context and content of the task carefully. "
        "**If the task is a classic puzzle, logic problem, or game, categorize it under 'Puzzles and Games'.** "
        "You must select the most appropriate domain and subdomain only from the list provided below. "
        "Do not provide any additional commentary or categories not listed.\n\n"
        "Respond in the exact format: 'Domain: [domain_name]; Subdomain: [subdomain_name]'.\n\n"
        f"{available_domains}"
    )

    print("\nCalculating the cost of the addition of doamin and subdomain columns using OpenAI GPT model:")
    total_cost = calculate_cost(MODEL_NAME, domain_subdomain_calculation_func, unique_task_df, system_prompt)

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


    root_dir = os.path.dirname(os.path.realpath(__file__))
    unique_tasks_domain_subdomain_path = os.path.join(root_dir, 'rosetta/unique_tasks_domain_subdomain.csv')

    if os.path.exists(unique_tasks_domain_subdomain_path):
        print("There is already a csv of the domain and subdomain to each unique task.\nWe will use it...")
        unique_task_with_domains = pd.read_csv(unique_tasks_domain_subdomain_path)
    else:
        print("No existing csv in folder. Proceeding with creating one...")
        unique_task_with_domains = identify_domains_and_subdomains(client, system_prompt, unique_task_df)

        # Save as a csv
        unique_task_with_domains.to_csv(unique_tasks_domain_subdomain_path, encoding='utf-8-sig', index=False)

    return unique_task_with_domains



       
def add_domains_to_rosetta(rosetta, unique_task_with_domains):

    root_dir = os.path.dirname(os.path.realpath(__file__))
    rosetta_path = os.path.join(root_dir, 'rosetta/rosetta.csv')
    rosetta.to_csv(rosetta_path, encoding='utf-8-sig', index=False, quoting=csv.QUOTE_ALL)

    rosetta_with_domains_subdomains = rosetta.copy()
    rosetta_with_domains_subdomains.insert(2, 'domain', None)
    rosetta_with_domains_subdomains.insert(3, 'subdomain', None)

    # Iterate over df and fill the domain and subdomain based on results_df
    for index, row in rosetta_with_domains_subdomains.iterrows():
        match = unique_task_with_domains[unique_task_with_domains['task'] == row['task']]
        if not match.empty:
            rosetta_with_domains_subdomains.at[index, 'domain'] = match.iloc[0]['domain']
            rosetta_with_domains_subdomains.at[index, 'subdomain'] = match.iloc[0]['subdomain']

    #     # Filter the DataFrame based on the task and language conditions
    # filtered_code = rosetta_with_domains_subdomains[(rosetta_with_domains_subdomains['task'] == 'AVL tree') & 
    #                                                 (rosetta_with_domains_subdomains['language'] == 'python')]['code']

    # # Calculate and print the number of characters in the filtered 'code' content
    # for idx, code in enumerate(filtered_code):
    #     print(f"-----------------------------Code snippet {idx + 1}: {len(code)} characters")

    root_dir = os.path.dirname(os.path.realpath(__file__))
    rosetta_with_domains_subdomains_path = os.path.join(root_dir, 'rosetta/rosetta_with_domains_subdomains.csv')
    rosetta_with_domains_subdomains.to_csv(rosetta_with_domains_subdomains_path, encoding='utf-8-sig', index=False, quoting=csv.QUOTE_NONNUMERIC)


    return rosetta_with_domains_subdomains