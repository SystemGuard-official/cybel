from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

# Load a SentenceTransformer model
embedding_model_name = "all-MiniLM-L6-v2"  # Choose a SentenceTransformers model
sentence_transformer = SentenceTransformer(embedding_model_name)

# Define a custom embedding function compatible with LangChain
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """Embed a list of documents and return as lists."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embed a single query and return as a list."""
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

embeddings = SentenceTransformerEmbeddings(sentence_transformer)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="my_documents",
    embedding_function=embeddings,
    persist_directory="./chromadb_persist"  # Path to persist data
)

# Example usage: Add documents to the vector store
# documents = ["python is a programming language", "java is a programming language"]
# metadata = [{"id": 1}, {"id": 2}]

# vector_store.add_texts(documents, metadatas=metadata)

# embed query
# semantic_search

# Perform semantic search
def semantic_search(query: str, top_k: int = 3):
    """Perform semantic search using the Chroma vector store."""
    results = vector_store.similarity_search(query, k=top_k)
    if not results:
        return None
    return [(result.page_content, result.metadata) for result in results]

# Example usage: Perform semantic search
query = "python"
results = semantic_search(query)
print(results)