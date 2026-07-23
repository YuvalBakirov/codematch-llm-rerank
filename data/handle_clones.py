
MODEL_NAME = "gpt-4o-mini"

def generate_clone_by_prompt(client, source_code, prompt):
    """
    Use OpenAI API to generate clones based on the provided prompt and source code.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": source_code}
            ]
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        print(f"Error generating types: {e}")
        return None


def create_clone(client, source_code, clone_type, task, rosetta_data):
    """
    Generate clones based on the clone type.
    """
    clone_codes = []
    if clone_type['clone_type'] == 'Type 1 Clone (Exact Clone)' and clone_type['clone_sub_type'] == "Identical Clone":
        clone_codes.append((source_code, 'python'))
    elif clone_type['clone_type'] == "Type 4 Clone (Semantic Clone)" and clone_type['clone_sub_type'] == "Different Language":
        filtered_data = rosetta_data[(rosetta_data['task'] == task) & (rosetta_data['language'] != 'python')][['code', 'language']]
        clone_codes = list(filtered_data.itertuples(index=False, name=None))
    else:
        clone_codes.append((generate_clone_by_prompt(client, source_code, clone_type['prompt']), 'python'))
    
    return clone_codes


def generate_clones(client, source_code, base_code_id, domain, subdomain, task, clone_types, rosetta_data, benchmark_test_data):
    # Create clones
    clone_codes = []
    for _, clone_type in clone_types.iterrows():
        clone_codes = create_clone(client, source_code, clone_type, task, rosetta_data)
        for clone_code in clone_codes:
            new_row = {
                'clone_code_id': f"{base_code_id}_{clone_type['clone_type_index']}",
                'base_code_id': base_code_id,
                'task': task,
                'domain': domain,
                'subdomain': subdomain,
                'clone_language': clone_code[1],
                'clone_type': f"T{clone_type['clone_type_index'][0]}",
                'clone_sub_type': clone_type['clone_sub_type'],
                'code': clone_code[0],
            }
            benchmark_test_data = benchmark_test_data.append(new_row, ignore_index=True)

    return benchmark_test_data


