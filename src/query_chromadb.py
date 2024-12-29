import os
from langchain_openai import OpenAI
from src.embedder import initialize_vector_store
from src.stopword_filter import filter_stopwords

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = ""


class OpenAITemperature:
    """Enum-like class for OpenAI temperature settings."""
    ZERO = 0.0
    LOW = 0.3
    MEDIUM = 0.7
    HIGH = 1.0


# Initialize the embedding model and LLM
try:
    llm = OpenAI(temperature=OpenAITemperature.LOW)
except Exception as e:
    raise RuntimeError("Failed to initialize the OpenAI LLM. Ensure API keys and configurations are correct.") from e

# Load the persisted Chroma vector store
try:
    EMBEDDING_TYPE = "sentence_transformers"  # Change to "openai" as needed
    VECTOR_STORE = initialize_vector_store(
        embedding_type=EMBEDDING_TYPE,
        collection_name="my_documents",
        persist_directory="./chromadb_persist"
    )
except Exception as e:
    raise RuntimeError("Failed to initialize the vector store. Check embeddings and persistence configuration.") from e


def semantic_search(query: str, top_k: int = 3):
    """
    Perform semantic search using the Chroma vector store.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of tuples containing the content and metadata of results.

    Raises:
        ValueError: If the query is empty after stop-word filtering.
    """
    # Filter stop words from the query
    cleaned_query = filter_stopwords(query)
    if not cleaned_query:
        raise ValueError("Query is empty after stop-word filtering.")
    
    # Perform similarity search
    results = VECTOR_STORE.similarity_search(cleaned_query, k=top_k)
    return [(result.page_content, result.metadata) for result in results] if results else []


def generate_response_with_context(query: str, rephrased_query: str, metadata: str, context: str):
    """
    Generate a response using LLM based on the query and retrieved context.

    Args:
        query (str): The original user query.
        rephrased_query (str): The rephrased query for better understanding.
        metadata (str): Metadata for references.
        context (str): Retrieved context for generating the response.

    Returns:
        str: The formatted response.
    """
    prompt = f"""
    System: You are an intelligent assistant creating a response to a query based on the provided context.

    Use the following context to answer the query. If you don't know the answer, say so.

    ## Tasks:
    1. Generate an accurate answer to the query using the context.
    2. Provide three relevant follow-up questions based on the query and context.
    3. Include references or citations from the metadata.

    ## Provided Context:
    {context}

    ## Metadata for Reference:
    {metadata}

    ## Query Details:
    Original Query: {query}
    Rephrased Query: {rephrased_query}

    ## Output Format:
    Answer:
    
    Follow-up Questions:
    1. 
    2.
    3.

    References:
    """

    return llm.invoke(prompt)


def rephrase_query(query: str) -> str:
    """
    Rephrase the user query for better clarity using LLM.

    Args:
        query (str): The original user query.

    Returns:
        str: The rephrased query.
    """
    prompt = f"""
    System: You are an expert in rephrasing queries.
    Please rephrase the following query for better clarity:

    Query:
    {query}
    """
    return llm.invoke(prompt)


def process_query(query: str) -> dict:
    """
    Process the user query by performing semantic search and generating a response.

    Args:
        query (str): The user query.

    Returns:
        dict: A dictionary containing the answer, follow-up questions, and references.
    """
    try:
        # Rephrase the query
        rephrased_query = rephrase_query(query)

        # Perform semantic search
        search_results = semantic_search(rephrased_query)
        if search_results:
            context = "\n".join(f"Context {idx}: {content}" for idx, (content, _) in enumerate(search_results))
            metadata = "\n".join(f"Metadata {idx}: {meta}" for idx, (_, meta) in enumerate(search_results))
        else:
            context = "No relevant context found."
            metadata = "No metadata available."

        # Generate response
        response = generate_response_with_context(query, rephrased_query, metadata, context)

        # Parse the response
        answer = response.split("Answer:")[1].split("Follow-up Questions:")[0].strip()
        follow_up_questions = response.split("Follow-up Questions:")[1].strip().split("\n")
        follow_up_questions = [q.strip() for q in follow_up_questions if q.strip()]

        return {"answer": answer, "follow_ups": follow_up_questions, "references": metadata}

    except ValueError as ve:
        return {"answer": str(ve), "follow_ups": [], "references": ""}
    except Exception as e:
        return {"answer": "An error occurred while processing the query.", "follow_ups": [], "references": ""}
