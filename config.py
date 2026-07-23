import os

import warnings
warnings.filterwarnings("ignore")


# File paths
#----------------------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ORIGINAL_CODE_CSV = "data/evaluation-datasets/original_code_benchmark_fixed.csv"
BENCHMARK_TEST_CODE_CSV = "data/evaluation-datasets/test_code_benchmark_fixed.csv"
ORIGINAL_CODE_PATH = os.path.join(ROOT_DIR, ORIGINAL_CODE_CSV)
BENCHMARK_TEST_PATH = os.path.join(ROOT_DIR, BENCHMARK_TEST_CODE_CSV)
#----------------------------------------------------------------------------------------


# Model settings
#----------------------------------------------------------------------------------------
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "huggingface_models/")
#----------------------------------------------------------------------------------------


# Available Models
#----------------------------------------------------------------------------------------
AVAILABLE_MODELS = ["codebert-base", "graphcodebert-base", "codet5-base", "Qwen2.5-Coder-0.5B", "Qwen2.5-Coder-0.5B-pe"]

MODEL_NAME = "microsoft/codebert-base"              # (Embedding size - 768     | Max token input size - 514)  # Default model
# Link - https://huggingface.co/microsoft/codebert-base

# MODEL_NAME = "microsoft/graphcodebert-base"       # (Embedding size - 50265   | Max token input size - 514)
# Link - https://huggingface.co/microsoft/graphcodebert-base

# MODEL_NAME = "Salesforce/codet5p-2b"              # (Embedding size - ?)
# Link - https://huggingface.co/Salesforce/codet5p-2b (Too heavy!! - reached 100% ssd when loading)

# MODEL_NAME = "Salesforce/codet5-base"             # (Embedding size - 32100   | Max token input size - 512)
# Link - https://huggingface.co/Salesforce/codet5-base 

# MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"            # (Embedding size - 151936  | Max token input size - 32768)
# Link - https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B
#----------------------------------------------------------------------------------------


# Qdrant settings
#----------------------------------------------------------------------------------------
def get_qdrant_host():
    """
    Determine the correct Qdrant host:
    - Use the `QDRANT_HOST` environment variable if provided.
    - Default to `localhost` when running locally.
    - Default to `qdrant` when running inside Docker.
    """
    # Check if running inside a Docker container
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            if 'docker' in f.read():
                return os.getenv("QDRANT_HOST", "qdrant")  # Default for Docker
    except FileNotFoundError:
        pass
    return os.getenv("QDRANT_HOST", "localhost")  # Default for local runs


# Qdrant settings
QDRANT_HOST = get_qdrant_host()
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_MAX_EMBEDDING_SIZE = 65536   # Qdrants' Maximum Storable Embedding Size
BENCHMARK_COLLECTION_NAME = "originals_embeddings"
#----------------------------------------------------------------------------------------


# Embeddings settings
#----------------------------------------------------------------------------------------
EMBEDDING_SIZE = None   # (initialized later during runtime)
DESIRED_EMBEDDING_SIZE = 768
REDUCE_EMBEDDING_SIZE = False
MAX_TOKEN_INPUT_SIZE = None
DESIRED_MAX_TOKEN_INPUT_SIZE = 3000



