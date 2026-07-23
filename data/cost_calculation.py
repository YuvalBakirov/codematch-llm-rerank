import tiktoken
import pandas as pd
import tqdm

MODEL_NAME = "gpt-4o-mini"

# Dictionary containing the cost per 1M tokens for input and output for various OpenAI models
model_costs = {
    "gpt-3.5-turbo": {"input_cost_per_1m_tokens": 0.002, "output_cost_per_1m_tokens": 0.002},
    "gpt-4o": {"input_cost_per_1m_tokens": 2.50, "output_cost_per_1m_tokens": 10.00},
    "gpt-4o-mini": {"input_cost_per_1m_tokens": 0.150, "output_cost_per_1m_tokens": 0.600}
}

# Function to estimate token count for a given text
def estimate_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

# Function to estimate token count for a given text
def estimate_tokens_by_model(text, model_name):

    # Get the tokenizer for the given model
    tokenizer = tiktoken.encoding_for_model(model_name)

    return len(tokenizer.encode(text))


# Function to calculate the cost of running a model with given prompts and a DataFrame
def calculate_cost(model_name, calculation_func, *args):
    if model_name not in model_costs:
        raise ValueError(f"Model {model_name} not found in cost dictionary.")

    # Get the tokenizer for the given model
    tokenizer = tiktoken.encoding_for_model(model_name)

    # Run the calculation function to get input and output tokens
    input_tokens, total_response_tokens = calculation_func(tokenizer, *args)

    # Get the cost per 1M tokens for the model
    input_cost_per_1m_tokens = model_costs[model_name]["input_cost_per_1m_tokens"]
    output_cost_per_1m_tokens = model_costs[model_name]["output_cost_per_1m_tokens"]

    # Calculate input and output costs
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m_tokens
    output_cost = (total_response_tokens / 1_000_000) * output_cost_per_1m_tokens

    # Total cost calculation
    total_cost = input_cost + output_cost

    # Print results
    print(f"Estimated total input token usage (including system prompt): {input_tokens}")
    print(f"Estimated total response token usage: {total_response_tokens}")
    print(f"Estimated total token usage: {input_tokens + total_response_tokens}")
    print(f"Estimated input cost: ${input_cost:.2f}")
    print(f"Estimated output cost: ${output_cost:.2f}")
    print(f"\nEstimated total cost: ${total_cost:.2f}")

    return total_cost




# Function to perform all calculations and return total input and output tokens
def domain_subdomain_calculation_func(tokenizer, *args):

    df, system_prompt = args

    user_prompt = "Task: {task}\nTask Description: {task_description}\nCode:\n{code}"

    # Estimate tokens for the system prompt
    system_prompt_tokens = estimate_tokens(system_prompt, tokenizer)

    # Calculate total input tokens including system prompt for each row
    input_tokens = system_prompt_tokens * len(df)
    for _, row in df.iterrows():
        full_user_prompt = user_prompt.format(task=row['task'], task_description=row['task_description'], code=row['code'])
        input_tokens += estimate_tokens(full_user_prompt, tokenizer)

    # Estimate response tokens (can be parameterized as needed)
    response_tokens_per_call = 15
    total_response_tokens = response_tokens_per_call * len(df)

    return input_tokens, total_response_tokens


# Function to perform all calculations and return total input and output tokens
def clones_calculation_func(tokenizer, *args):

    df, system_prompt, source_code = args

    # Estimate tokens for the system prompt
    system_prompt_tokens = estimate_tokens(system_prompt, tokenizer)

    # Calculate total input tokens including system prompt for each row
    input_tokens = system_prompt_tokens * 11 * len(df)

    for _, row in df.iterrows():
        # full_user_prompt = user_prompt.format(task=row['task'], task_description=row['task_description'], code=row['code'])
        input_tokens += estimate_tokens(source_code, tokenizer) * 11

    # Estimate response tokens (can be parameterized as needed)
    response_tokens_per_call = estimate_tokens(source_code, tokenizer)
    total_response_tokens = response_tokens_per_call * len(df) * 11

    return input_tokens, total_response_tokens


# Function to perform all calculations and return total input and output tokens
def clones_calculation_avg_func(tokenizer, *args):

    df, system_prompt, avg_input_tokens = args

    # Estimate tokens for the system prompt
    system_prompt_tokens = estimate_tokens(system_prompt, tokenizer)

    # Calculate total input tokens including system prompt for each row
    input_tokens = system_prompt_tokens * 11 * len(df)

    for _, row in df.iterrows():
        # full_user_prompt = user_prompt.format(task=row['task'], task_description=row['task_description'], code=row['code'])
        input_tokens += estimate_tokens(row['code'], tokenizer) * 11

    total_response_tokens = 0

    for _, row in df.iterrows():
        # Estimate response tokens (can be parameterized as needed)
        total_response_tokens += estimate_tokens(row['code'], tokenizer) * 11

    # total_response_tokens = response_tokens_per_call * len(df) * 11

    return input_tokens, total_response_tokens



def identify_domain_subdomain(client, system_prompt, task, task_description, code):
    """
    Use OpenAI API to identify the domain and subdomain that best represent the given code.
    Sends the full list of available domains and subdomains in each prompt for consistency.
    """

    user_prompt = (
        f"Task: {task}\n"
        f"Task Description: {task_description}\n"
        f"Code:\n{code}"
    )

    try:

        completion = client.chat.completions.create(
            model=MODEL_NAME,  # Ensure this is the correct model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        temperature=0  # Set temperature to 0 for deterministic output
        )

        response = completion.choices[0].message.content.strip()
        return response
    
    except Exception as e:
        print(f"Error identifying domain and subdomain: {e}")
        return None


def identify_domains_and_subdomains(client, system_prompt, unique_task_df):
    # Create a new DataFrame to store the results
    results_df = pd.DataFrame(columns=['task', 'domain', 'subdomain'])

    # Use tqdm to display progress bar
    for _, row in tqdm(unique_task_df.iterrows(), total=len(unique_task_df)):
        response = identify_domain_subdomain(client, system_prompt, row['task'], row['task_description'], row['code'])
        
        if response:
            # Parsing the response to extract domain and subdomain
            if 'Domain:' in response and 'Subdomain:' in response:
                domain_start = response.find('Domain:') + len('Domain: ')
                domain_end = response.find('; Subdomain:')
                subdomain_start = domain_end + len('; Subdomain: ')
                
                domain = response[domain_start:domain_end].strip()
                subdomain = response[subdomain_start:].strip()

                # Append the results to the DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([{
                    'task': row['task'],
                    'domain': domain,
                    'subdomain': subdomain
                }])], ignore_index=True)

    return results_df


def calculate_clones_creation(unique_task_df):

    print(f"\nCalculating the cost of adding 11 clones to {unique_task_df.shape[0]} base codes using OpenAI GPT model:\n")
    system_prompt = "Provide an exact copy of the given code. Return only the code without any comments, explanations, or formatting."
    total_cost = calculate_cost(MODEL_NAME, clones_calculation_avg_func, unique_task_df, system_prompt, 345)
