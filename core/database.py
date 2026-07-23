import pandas as pd

from tqdm import tqdm

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from core.utils import generate_code_embedding_generic

import config
import logging


def initialize_client(logger):
    try:
        client = QdrantClient(url=f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}")
        logger.info("Qdrant client initialized successfully.")
        return client
    except Exception as e:
        exp = f"Failed to initialize Qdrant client: {e}"
        logger.error(exp)
        raise Exception(exp)
        

def init_vector_db(client, embedding, collection_name, index, id, metadata):
    """
    Initialize the vector database with the given embedding and metadata.

    Args:
        embedding (numpy.ndarray): The embedding to store in the vector database.
        index (str): The index of the embedding.
        id (int): The id of the embedding.
        metadata (dict): The metadata associated with the embedding.
    """
    payload = {"base_code_id": index, **metadata}
    point = PointStruct(id=id, vector=embedding.tolist(), payload=payload)
    client.upsert(collection_name=collection_name, points=[point])
    # print(f"Initialized embedding for index {index} with metadata into the vector database.")


def create_collection(logger, client, collection_name):
    try:
        client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=config.EMBEDDING_SIZE,
                        distance=models.Distance.COSINE
                    ),
                )
        logger.info(f"Collection '{collection_name}' created successfully.")

    except Exception as e:
        exp = f"Failed to create a collection in the vector database: {e}"
        logger.error(exp)
        raise Exception(exp)
        

def handle_vectordb_creation(client, collection_name, data_path_to_populate, model, tokenizer):
    """
    Handles the creation and population of a vector database collection.
    
    If the specified collection already exists, the user is prompted to decide whether
    to clear and refill the database with fresh data. Otherwise, a new collection is created
    and populated with embeddings.

    Args:
        client: Vector database client instance.
        collection_name (str): Name of the collection to check or create.
        data_path_to_populate (str): Path to the data file for populating the database.
        model: Model used for embedding generation.
        tokenizer: Tokenizer used for preprocessing the code before embedding.
    """

    vectordb_logger = logging.getLogger("benchmark.vectordb")

    # Check if the collection already exists
    if client.collection_exists(collection_name=collection_name):

        collection_info = client.get_collection(collection_name=collection_name)
        # Get embedding size (vector size)
        embedding_size = collection_info.config.params.vectors.size

        vectordb_logger.info(f"Collection '{collection_name}' (with embedding size - {embedding_size}) already exists in the vector database.")
        
        while True:
            response = input("\nDo you want to clear and refill the collection? (y/n): ").strip().lower()
            if response == 'y':
                vectordb_logger.info(f"Proceeding to clear and refill collection '{collection_name}'...")
                clear_vector_db(vectordb_logger,client, [collection_name])
                create_collection(vectordb_logger, client, collection_name)
                populate_vector_db(vectordb_logger, client, data_path_to_populate, collection_name, model, tokenizer)
                vectordb_logger.info(f"Collection '{collection_name}' has been successfully refilled.")
                break
            elif response == 'n':
                vectordb_logger.info(f"Collection '{collection_name}' remains unchanged.")
                break
            else:
                print("Invalid input. Please enter 'y' (yes) or 'n' (no).")
    else:
        vectordb_logger.info(f"Collection '{collection_name}' does not exist. Creating and populating it...")
        create_collection(vectordb_logger, client, collection_name)
        populate_vector_db(vectordb_logger, client, data_path_to_populate, collection_name, model, tokenizer)
        vectordb_logger.info(f"Collection '{collection_name}' has been successfully created and populated.")
        

def clear_vector_db(logger, client, collection_names=None):
    """
    Clear the vector database by deleting the collection.
    """
    try:
        
        if collection_names is None:
            logger.info("Clearing the whole DB...")
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]

        for collection_name in collection_names:
            if client.collection_exists(collection_name=collection_name):
                client.delete_collection(collection_name=collection_name)
                logger.info(f"Collection '{collection_name}' deleted successfully.")
            else:
                logger.info(f"Collection '{collection_name}' does not exist. Skipping deletion.")

    except Exception as e:
        exp = f"Failed to clear the vector database: {e}"
        logger.error(exp)
        raise Exception(exp)
        


def populate_vector_db(logger, client, csv_file, collection_name, model, tokenizer):
    """
    Index original open source code from the CSV file into the vector database with specific metadata.

    Args:
        csv_file (str): Path to the CSV file containing code examples.
        model_name (str): The name of the model to use for embedding creation.
    """
    try:

        df = pd.read_csv(csv_file)
        for id, row in tqdm(df.iterrows(), desc="Populating Vector Database...", total=len(df)):
            code = str(row['code']).strip()
            if not code:
                # print(f"Skipping ID {id}: code is empty or invalid.")
                logger.warning(f"Skipping ID {id}: code is empty or invalid.")
                continue
            
            base_code_id = row['base_code_id']
            
            # embedding = generate_code_embedding(code, model, tokenizer)
            embedding = generate_code_embedding_generic(logger, code, model, tokenizer)

            if embedding is not None:
                metadata = {
                    'language': row['language'],
                    'task': row['task']
                }
                
                if 'domain' in row:
                    metadata['domain'] = row['domain']
                if 'subdomain' in row:
                    metadata['subdomain'] = row['subdomain']

                init_vector_db(client, embedding, collection_name, base_code_id, id, metadata)
                # print(f"ID {id}, Base Code ID: {base_code_id}")

    except Exception as e:
        exp = f"Failed to populate the vector database: {e}"
        logger.error(exp)
        raise Exception(exp)
        