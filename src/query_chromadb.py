import os
os.environ["OPENAI_API_KEY"] = ""

from langchain_openai import OpenAI
from src.embedder import initialize_vector_store
class OpenAITemperature:
    """Enum-like class for OpenAI temperature settings."""
    ZERO = 0.0
    LOW = 0.3
    MEDIUM = 0.7
    HIGH = 1.0

# Initialize the embedding model and LLM
llm = OpenAI(temperature=OpenAITemperature.MEDIUM)

# Load the persisted Chroma vector store
embedding_type = "sentence_transformers"  # Change to "openai" as needed
vector_store = initialize_vector_store(
    embedding_type=embedding_type,
    collection_name="my_documents",
    persist_directory="./chromadb_persist"
)
# Perform semantic search
def semantic_search(query: str, top_k: int = 3):
    """Perform semantic search using the Chroma vector store."""
    results = vector_store.similarity_search(query, k=top_k)
    if not results:
        return None
    return [(result.page_content, result.metadata) for result in results]

# Generate response using LLM
def generate_response_from_llm(query: str, rephrased_query: str, context):
    """Generate a response from the LLM using the search results as context."""
  
    # Create improved prompt
    prompt = f"""
    System: You are an intelligent assistant creating a response to a query based on the provided context.
    
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    
    ## Tasks:
    1. Use the provided context to generate an accurate answer to the query.
    2. Generate three follow-up questions based on the query and context.

    ## Instructions:
    - Use only explicit internal context information to generate the answer.
    - Response should be in the form of a paragraph.
    - Please formulate the answer the query based on the context provided.
    - Please formulate the follow  up questions based on the context provided.
    - Remember, you shall complete ALL tasks.
    - If the provided context contains contradictory or conflicting information, 
    - state so providing the conflicting information.

    ### Context:
    {context}

    ### Original Query:
    {query}
    
    ### Rephrased Query:
    {rephrased_query}

    ### Output Format:
    Answer of the query->
    
    Follow-up Questions:
    1. 
    2.
    3.
    """

    # Pass the prompt to the LLM
    response = llm.invoke(prompt)
    return response

# Rephrase query using LLM
def rephrase_query_with_langchain(query: str) -> str:
    """Rephrase the input query using the LLM."""
    rephrase_prompt = f"""
    System: You are an expert in rephrasing queries. 
    Please rephrase the following query clearly and concisely.

    Query:
    {query}
    """
    return llm.invoke(rephrase_prompt)


def process_response_for_api(query_text):
    try:
        # rephrased_query = rephrase_query_with_langchain(query_text)
        rephrased_query = query_text

        semantic_search_list = semantic_search(rephrased_query)

        # apply logic based on metadata to get previous and next chunk as well
        # there is a chance we matched with the middle of the document
        # we will require the previous and next chunk to get the full context

        # Generate response using LLM
        print("Generating response...\n")
        if semantic_search_list:
            context = "\n".join((f"context {idx}: {content}" for idx, (content, _) in enumerate(semantic_search_list)))
        else:
            context = "No relevant context found."

        print(f"Context:\n{context}\n")
        # response = generate_response_from_llm(query_text, rephrased_query, context)
        
        # # formulate the output
        # answer = response.split("Answer of the query->")[1].split("Follow-up Questions:")[0].strip()
        # follow_ups = response.split("Follow-up Questions:")[1].strip().split("\n")
        # follow_ups = [f.strip() for f in follow_ups if f.strip()]
        return {"answer": context, 
                "follow_ups": ["What is AI?", "How is AI used in everyday life?", "What are the different types of AI?"]}
    except Exception as e:
        print(f"Error processing response: {e}")
        return {"answer": "Error processing response.", "follow_ups": []}

    
 

# # Main function
# if __name__ == "__main__":
#     process_response()