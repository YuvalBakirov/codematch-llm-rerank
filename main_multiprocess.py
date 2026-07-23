import config
import time
import uuid
import logging
import multiprocessing

from core.utils import start_qdrant_server, load_huggingface_model, get_embedding_and_token_size
from core.database import initialize_client, handle_vectordb_creation
from core.metrics import direct_clone_comparison_test, global_clone_search

# Import the multi-process logging setup
from core.mylogger import setup_logging_benchmark_multiprocess


def run_direct_clone_comparison(benchmark_path, model_name):
    """
    Child process entry point for Direct Clone Comparison.
    Re-initialize logging so child logs also go to the multi-process files.
    """
    setup_logging_benchmark_multiprocess()
    child_logger = logging.getLogger("benchmark.direct_comparison")

    child_logger.info("Initializing child process for Direct Clone Comparison...")
    client = initialize_client(child_logger)
    model, tokenizer = load_huggingface_model(child_logger, config.MODEL_NAME, config.MODEL_CACHE_DIR)
    config.EMBEDDING_SIZE, config.MAX_TOKEN_INPUT_SIZE = get_embedding_and_token_size(child_logger, model, tokenizer)

    child_logger.info("Starting Direct Clone Comparison...")
    direct_clone_comparison_test(client, benchmark_path, model, tokenizer, model_name)
    child_logger.info("Direct Clone Comparison completed.")


def run_global_clone_search(benchmark_path, model_name):
    """
    Child process entry point for Global Clone Search.
    Re-initialize logging so child logs also go to the multi-process files.
    """
    setup_logging_benchmark_multiprocess()
    child_logger = logging.getLogger("benchmark.global_search")

    child_logger.info("Initializing child process for Global Clone Search...")
    client = initialize_client(child_logger)
    model, tokenizer = load_huggingface_model(child_logger, config.MODEL_NAME, config.MODEL_CACHE_DIR)
    config.EMBEDDING_SIZE, config.MAX_TOKEN_INPUT_SIZE = get_embedding_and_token_size(child_logger, model, tokenizer)

    child_logger.info("Starting Global Clone Search...")
    global_clone_search(client, benchmark_path, model, tokenizer, model_name)
    child_logger.info("Global Clone Search completed.")


def main():
    """
    Main process entry point for the multi-process run.
    """
    # 1) Set up logging in the main process for multi-process mode
    setup_logging_benchmark_multiprocess()
    benchmark_logger = logging.getLogger("benchmark")

    run_id = uuid.uuid4()
    benchmark_logger.info("=" * 50)
    benchmark_logger.info(f"New Multi-Process Run Started | Run ID: {run_id}")
    benchmark_logger.info("=" * 50)

    start_time = time.time()

    # 2) Basic environment/log info
    benchmark_logger.info(f"Number of CPUs: {multiprocessing.cpu_count()}")
    benchmark_logger.info(f"Using model: {config.MODEL_NAME}")

    # 3) Load model + tokenizer in the main process (optional pre-loading)
    model, tokenizer = load_huggingface_model(benchmark_logger, config.MODEL_NAME, config.MODEL_CACHE_DIR)
    config.EMBEDDING_SIZE, config.MAX_TOKEN_INPUT_SIZE = get_embedding_and_token_size(benchmark_logger, model, tokenizer)

    benchmark_logger.info(f"Embedding size for model '{config.MODEL_NAME}': {config.EMBEDDING_SIZE}")
    benchmark_logger.info(f"Maximum token input size: {config.MAX_TOKEN_INPUT_SIZE}")

    # 4) Start Qdrant server and initialize DB in the main process
    start_qdrant_server(benchmark_logger)
    client = initialize_client(benchmark_logger)
    handle_vectordb_creation(
        client=client,
        collection_name=config.BENCHMARK_COLLECTION_NAME,
        data_path_to_populate=config.ORIGINAL_CODE_PATH,
        model=model,
        tokenizer=tokenizer
    )

    # 5) Prepare for child processes
    benchmark_path = config.BENCHMARK_TEST_PATH
    model_name = config.MODEL_NAME.split('/')[1]
    benchmark_logger.info(f"Launching Direct Clone Comparison and Global Clone Search in parallel...")

    # 6) Create child processes
    p1 = multiprocessing.Process(target=run_direct_clone_comparison, args=(benchmark_path, model_name))
    p2 = multiprocessing.Process(target=run_global_clone_search, args=(benchmark_path, model_name))

    # 7) Start and join
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    benchmark_logger.info("Both tests completed successfully.")

    total_time = (time.time() - start_time) / 60
    benchmark_logger.info(f"Run ID: {run_id} | Total time taken: {total_time:.2f} minutes")
    benchmark_logger.info("=" * 50)


if __name__ == '__main__':
    main()
