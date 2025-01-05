import os
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Custom embedding class for SentenceTransformers
class SentenceTransformerEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Embed a list of documents and return as lists."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embed a single query and return as a list."""
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Function to initialize Chroma with the chosen embedding model
def initialize_vector_store(embedding_type, collection_name, persist_directory):
    """
    Initialize the Chroma vector store with the chosen embedding model.
    
    Args:
        embedding_type (str): "sentence_transformers" or "openai"
        collection_name (str): Name of the Chroma collection.
        persist_directory (str): Path for persistence.

    Returns:
        Chroma: The initialized vector store.
    """
    # models_names = ["all-MiniLM-L6-v2", "all-MPNet-base-v2", "sentence-t5-xxl",
    #                 "paraphrase-multilingual-MiniLM-L12-v2", ]

    if embedding_type == "sentence_transformers":
        model_name = "all-MPNet-base-v2"
        embedding_function = SentenceTransformerEmbeddings(model_name)
    elif embedding_type == "openai":
        model_name = "text-embedding-3-small"
        embedding_function = OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError("Invalid embedding type. Choose 'sentence_transformers' or 'openai'.")

    # Initialize Chroma vector store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function, # type: ignore
        persist_directory=persist_directory
    )
    return vector_store

