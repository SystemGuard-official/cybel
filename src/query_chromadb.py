import os
import random
from langchain_openai import OpenAI
from src.embedder import initialize_vector_store
from src.stopword_filter import filter_stopwords

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


EMBEDDING_TYPE = "sentence_transformers"  # Change to "openai" as needed
document_collection = "my_documents"
persist_directory = "./chromadb_persist"


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
    print(f"Error initializing OpenAI model: {e}")

# Load the persisted Chroma vector store
try:
    VECTOR_STORE = initialize_vector_store(
        embedding_type=EMBEDDING_TYPE,
        collection_name=document_collection,
        persist_directory=persist_directory
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


def generate_response_with_context(query: str, retrieved_documents: str, metadata: str):
    """
    Generate a response using LLM based on the query and retrieved context.

    Args:
        query (str): The original user query.
        retrieved_documents (str): Retrieved context for generating the response.
        metadata (str): Metadata for references.

    Returns:
        str: The formatted response.
    """
    prompt = f"""
    You are tasked with answering a question based STRICTLY on the provided context. you have to formulate the answer and write in your own words.
        
    Rules and Guidelines:
    1. only use information from the provided context documents to formulate your answer and follow-up questions.
    2. Follow up question should be connected to the context and should be relevant to the query. follow  up questions is generally created question based on the context to get more information.
    2. If the answer cannot be fully derived from the context, state "Cannot provide a complete answer based on available context"
    3. Do not make assumptions or use external knowledge to answer the query.
    5. Response should be in the form of a paragraph or a list of key points or table if required
    6. create References section with the sources used to answer the query or generate follow-up questions, N/A if not applicable
    7. Return the output answer in proper markdown format for easy reading and interpretation use headings and bullet points where necessary

    Use the following retrieved documents to answer the query or generate follow-up questions. If the answer is not in the documents, respond with "I don't know."

    Retrieved Documents:
    {retrieved_documents}

    ### Query:
    {query}

    ### Metadata: to create references and sources
    {metadata}
    
    ### Output Format:
    Answer:
    
    Follow-up Questions:
    1. 
    2.
    3.

    References:
    1.
    2.
    """

    # Pass the prompt to the LLM
    response = llm.invoke(prompt)
    return response


def rephrase_query(query: str) -> str:
    """
    Rephrase the user query for better clarity using LLM.

    Args:
        query (str): The original user query.

    Returns:
        str: The rephrased query.
    """
    prompt = f"""
    System: You are an expert in rephrasing queries. Please rephrase the following query for better clarity:

    Query:
    {query}
    """
    return llm.invoke(prompt)


def process_query(query: str, number_of_results: int = 3, is_rephrased: bool = False):
    """
    Process the user query by performing semantic search and generating a response.

    Args:
        query (str): The user query.

    Returns:
        dict: A dictionary containing the answer, follow-up questions, and references.
    """
    # Rephrase the query
    if is_rephrased:
        query = rephrase_query(query)

    # Perform semantic search to retrieve context
    search_results = semantic_search(query, top_k=number_of_results)
    if search_results:
        context = "\n".join(f"Context {idx}: {content}" for idx, (content, _) in enumerate(search_results))
        metadata = "\n".join(f"Metadata {idx}: {meta}" for idx, (_, meta) in enumerate(search_results))
    else:
        context = "No relevant context found."
        metadata = "No metadata available."

    # Generate response
    response = generate_response_with_context(query, context, metadata)

    # Parse the response
    answer = response.split("Answer:")[1].split("Follow-up Questions:")[0].strip()

    try:
        follow_up_questions = response.split("Follow-up Questions:")[1].split("References:")[0].strip().split("\n")
        follow_up_questions = [q.strip() for q in follow_up_questions if q.strip()]
    except IndexError:
        follow_up_questions = []

    try:
        references = response.split("References:")[1].strip()
        references = [ref.strip() for ref in references.split("\n") if ref.strip()]
    except IndexError:
        references = []

    for idx, context in enumerate(search_results):
        print(f"Context {idx}: {context}")
    

    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"Follow-up Questions: {follow_up_questions}")
    print(f"References: {references}")

    return {"answer": answer, "follow_ups": follow_up_questions, 
            "references": references,
    } 


def get_random_document_chunks():
    """
    Retrieve three random chunks from the Chroma vector store.

    Returns:
        list: A list of tuples containing the content and metadata of random chunks.
    """
    # Ensure the vector store is initialized
    initial_questions = [
        "Hi, Tell me about yourself?",
        "Who built you?",
        "What programming language are you written in?",
    ]
    
    if VECTOR_STORE is None:
        raise RuntimeError("Vector store is not initialized.")

    # Get all the keys (IDs) of documents from the vector store
    try:
        all_keys = VECTOR_STORE._collection.get()["ids"]
    except Exception as e:
        return initial_questions

    if not all_keys:
        return initial_questions

    # Pick 3 random document IDs
    random_keys = random.sample(all_keys, min(3, len(all_keys)))

    # Retrieve documents by keys
    random_chunks = ""
    for key in random_keys:
        try:
            document = VECTOR_STORE._collection.get(ids=[key])
            if document["documents"]:
                print(f"Lenght of document: {len(document['documents'])}")
                content = document["documents"][0]  # Assuming one document per ID
            else:
                content = "No content available."
            random_chunks += f"{content}\n\n"
        except Exception as e:
            print(f"Failed to retrieve document for ID {key}: {e}")
    print(f"Random Chunks: {random_chunks}")

    # call llm to form question based on the context
    prompt = f"""
    Genearet 3 questions based on the following context. 
    
    Rules and Guidelines:
        - The questions should be relevant to the context and should be connected to the context.
        - you have to generate 3 questions based on the context provided and there should be answerable questions in the context and don't ask questions that are not answerable from the context.
        - don't use external knowledge to generate questions.
        - The questions should be in the form of a question and should be relevant to the context.
        - The questions should be clear and easy to understand.
        - The questions should be relevant to the context and should be connected to the context.

    Context:
    {random_chunks}

    Output Format:
    1. Question 1:
    2. Question 2:
    3. Question 3:
    """

    initial_questions = llm.invoke(prompt)
    # print(f"Random Chunks: {random_chunks}")
    # print(f"Generated Questions: {initial_questions}")

    # split and make a list of questions
    initial_questions = initial_questions.split("\n")
    initial_questions = [q.strip() for q in initial_questions if q.strip()]
    return initial_questions
