import config
import time
from core.utils import start_qdrant_server, load_huggingface_model, get_embedding_and_token_size
from core.database import initialize_client, handle_vectordb_creation
from core.metrics import direct_clone_comparison_test, global_clone_search
import uuid

import logging
from core.mylogger import setup_logging_benchmark

def main():
    setup_logging_benchmark()

    benchmark_logger = logging.getLogger("benchmark")

    run_id = uuid.uuid4()
    benchmark_logger.info("=" * 50)
    benchmark_logger.info(f"New Run Started | Run ID: {run_id}")
    benchmark_logger.info("=" * 50)

    benchmark_logger.info("Benchmarking process started")

    start_time = time.time()

    import multiprocessing
    benchmark_logger.info(f"Number of CPUs: {multiprocessing.cpu_count()}")

    # Load the model
    benchmark_logger.info(f"Using model: {config.MODEL_NAME}")
    model, tokenizer = load_huggingface_model(benchmark_logger, config.MODEL_NAME, config.MODEL_CACHE_DIR)

    benchmark_logger.info(f"Tokenizer name: {tokenizer.name_or_path}")
    benchmark_logger.info(f"Tokenizer class: {tokenizer.__class__.__name__}")

    # Get the embedding size
    config.EMBEDDING_SIZE, config.MAX_TOKEN_INPUT_SIZE = get_embedding_and_token_size(benchmark_logger, model, tokenizer)
    benchmark_logger.info(f"Embedding size for model '{config.MODEL_NAME}': {config.EMBEDDING_SIZE}")
    benchmark_logger.info(f"Maximum token input size: {config.MAX_TOKEN_INPUT_SIZE}")

    start_qdrant_server(benchmark_logger)

    client = initialize_client(benchmark_logger)

    # Initialize the vector database
    benchmark_logger.info("Initializing the vector database.")
    handle_vectordb_creation(
        client=client,
        collection_name=config.BENCHMARK_COLLECTION_NAME,
        data_path_to_populate=config.ORIGINAL_CODE_PATH,
        model=model,
        tokenizer=tokenizer
    )

    # Run tests
    benchmark_logger.info("Running tests:")

    benchmark_logger.info(f"1. Direct Clone Comparison Test")
    direct_clone_comparison_test(client, config.BENCHMARK_TEST_PATH, model, tokenizer, config.MODEL_NAME.split('/')[1])

    benchmark_logger.info(f"2. Global Clone Search")
    global_clone_search(client, config.BENCHMARK_TEST_PATH, model, tokenizer, config.MODEL_NAME.split('/')[1])

    total_time = (time.time() - start_time) / 60
    benchmark_logger.info("=" * 50)
    benchmark_logger.info(f"Run ID: {run_id} | Total time taken: {total_time:.2f} minutes")
    benchmark_logger.info("=" * 50)

if __name__ == '__main__':
    main()