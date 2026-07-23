import subprocess
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM, BitsAndBytesConfig

import config


def start_qdrant_server(logger):
    """
    Start the Qdrant server using Docker in a new terminal window.
    """
    docker_command = (
        'docker run -p 6333:6333 -p 6335:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant'
    )
    try:
        subprocess.Popen(['start', 'cmd', '/c', docker_command], shell=True)
        logger.info("Qdrant server started successfully.")
    except subprocess.CalledProcessError as e:
        exp = f"An error occurred while starting the Qdrant server: {e}"
        logger.error(exp)
            

def load_huggingface_model(logger, model_name, cache_dir=None):
    """
    Load a Hugging Face model and its tokenizer by model name.

    Args:
        model_name (str): The name of the model to load from the Hugging Face model hub.
        cache_dir (str, optional): Directory to cache and load models from. Defaults to None.

    Returns:
        model, tokenizer: Loaded Hugging Face model and tokenizer.
    """
    try:

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if model_name == "microsoft/codebert-base":
            # Load the model
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        if model_name in  ["Salesforce/codet5p-2b", "Salesforce/codet5-base"]:
            # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"
            )

        if model_name == "microsoft/graphcodebert-base":
            model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)

        if model_name == "Qwen/Qwen2.5-Coder-0.5B":
            # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"  # Automatically maps layers to available devices
            )

        logger.info(f"Successfully loaded model and tokenizer: {model_name}")
        return model, tokenizer
    except Exception as e:
        exp = f"Failed to load model '{model_name}': {e}"
        logger.error(exp)
        raise RuntimeError(exp)


def get_embedding_and_token_size(benchmark_logger, model, tokenizer):
    """
    Determine the embedding size and maximum token input size of the Hugging Face model.

    Args:
        model: The Hugging Face model.
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        tuple: The embedding size and the maximum token input size.
    """
    # Create a dummy input
    dummy_input = tokenizer("dummy text", return_tensors="pt")
    
    # Check if the model requires both input_ids and decoder_input_ids
    if model.config.is_encoder_decoder:
        dummy_input["decoder_input_ids"] = dummy_input["input_ids"]
    
    # Pass the dummy input through the model
    outputs = model(**dummy_input)

    # Handle different output types
    if hasattr(outputs, "last_hidden_state"):  # Standard models
        embedding_size = outputs.last_hidden_state.size(-1)
    elif hasattr(outputs, "logits"):  # For masked LM models
        embedding_size = outputs.logits.size(-1)
    elif hasattr(outputs, "hidden_states"):  # Models with hidden states
        embedding_size = outputs.hidden_states[-1].size(-1)
    else:
        txt = "Model output does not contain embeddings in a known attribute."
        benchmark_logger.error(txt)
        raise AttributeError(txt)
    
    # # Retrieve the maximum token input size
    # max_token_input_size = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else None

    # Retrieve the maximum token input size
    max_token_input_size = getattr(model.config, 'max_position_embeddings', None)

    if max_token_input_size is None:
        # Fallback to tokenizer model_max_length
        max_token_input_size = getattr(tokenizer, 'model_max_length', None)

    # Validate the embedding size
    if embedding_size > config.QDRANT_MAX_EMBEDDING_SIZE:
        benchmark_logger.info(
            f"Warning: Model's embedding size ({embedding_size}) exceeds the maximum allowed size ({config.QDRANT_MAX_EMBEDDING_SIZE}). "
            f"Using the maximum allowed size instead."
        )
        embedding_size = config.QDRANT_MAX_EMBEDDING_SIZE
        # embedding_size = config.DESIRED_EMBEDDING_SIZE
        config.REDUCE_EMBEDDING_SIZE = True

    return embedding_size, max_token_input_size


def reduce_embedding_size(embedding, target_size = 768):
    """
    Truncate the embedding to the target size if it exceeds the limit.

    Args:
        embedding (numpy.ndarray): Original high-dimensional embedding.

    Returns:
        numpy.ndarray: Truncated embedding.
    """
    return embedding[:target_size] if embedding.shape[0] > target_size else embedding


def generate_code_embedding_generic(logger, code, model, tokenizer):
    """
    Create an embedding for the given code using the specified model with mixed precision.

    Args:
        code (str): The code snippet to create an embedding for.
        model: The Hugging Face model.
        tokenizer: The tokenizer for the model.

    Returns:
        numpy.ndarray: The embedding of the code snippet.
    """

    # max_length = model.config.max_position_embeddings

    # Tokenize the input code and move inputs to the device
    # inputs = tokenizer(
    #     code[:max_length],  # Truncate input if necessary 
    #     return_tensors="pt", 
    #     truncation=True, 
    #     padding=True, 
    #     max_length=config.MODEL_MAX_INPUT_TOKENS
    # )

    # Define the semantic prompt
    PROMPT = "You are a programming expert. Analyze the following code snippet semantically, focusing on its functionality and behavior. Understand its purpose, logic, and how it operates, while ignoring irrelevant details such as formatting or variable names:\nCode:\n"
    
    # Combine the prompt with the code
    full_input = PROMPT + code

    # Tokenize with truncation
    tokens = tokenizer.encode(full_input, truncation=True, max_length=config.MAX_TOKEN_INPUT_SIZE)

    # Calculate token size
    token_size = len(tokens)
    logger.info(f"Token size of full_input: {token_size}")
    logger.info(f"Token max input {config.MAX_TOKEN_INPUT_SIZE}")

    inputs = tokenizer(
        full_input,
        return_tensors="pt", 
        truncation=True, 
        max_length=config.MAX_TOKEN_INPUT_SIZE-2
    )

    # inputs = tokenizer(
    #     full_input,
    #     return_tensors="pt", 
    #     truncation=True, 
    #     padding=True,
    #     max_length=config.MAX_TOKEN_INPUT_SIZE
    # )

    # inputs = tokenizer(
    #     code,
    #     return_tensors="pt", 
    #     truncation=True, 
    #     padding=True
    # )


    # Handle encoder-decoder models (e.g., T5 or Bart)
    if model.config.is_encoder_decoder:
        inputs["decoder_input_ids"] = inputs["input_ids"]


    try:
        outputs = model(**inputs)
    except IndexError as e:
        exp = f"Error: {e}. Skipping input: {full_input[:100]}..."
        logger.error(exp)
        return None  # Skip this input

    # Extract embeddings based on model's output attributes
    if hasattr(outputs, "last_hidden_state"):  # Standard models (e.g., CodeBERT)
        embeddings = outputs.last_hidden_state
    elif hasattr(outputs, "logits"):  # Models outputting logits
        embeddings = outputs.logits
    elif hasattr(outputs, "hidden_states"):  # Models with intermediate hidden states
        embeddings = outputs.hidden_states[-1]  # Use the last layer
    else:
        txt = f"Model output does not contain embeddings in a known attribute. Skipping input: {full_input[:100]}..."
        logger.error(txt)
        raise AttributeError(txt)

    # Compute the mean embedding for the sequence
    embedding = embeddings.mean(dim=1).squeeze().detach().cpu().numpy()

    # Optional: Reduce embedding size for efficiency
    if hasattr(config, "REDUCE_EMBEDDING_SIZE") and config.REDUCE_EMBEDDING_SIZE:
        embedding = reduce_embedding_size(embedding, config.EMBEDDING_SIZE)

    return embedding


def clean_code(code):
    """
    Clean the code snippet by removing extra new lines and spaces.

    Args:
        code (str): The code snippet to clean.

    Returns:
        str: The cleaned code snippet.
    """
    return ' '.join(code.split())
