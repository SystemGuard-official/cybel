import os
import re
os.environ["OPENAI_API_KEY"] = ""

import json
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings import initialize_vector_store

# Initialize the embedding model
embedding_model_name = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=embedding_model_name)

# Initialize the Chroma vector store (persistent mode)
embedding_type = "sentence_transformers"  # Change to "openai" as needed
vector_store = initialize_vector_store(
    embedding_type=embedding_type,
    collection_name="my_documents",
    persist_directory="./chromadb_persist"
)

# Function to split large text into chunks
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store file content in ChromaDB with enhanced metadata
def store_file_in_chromadb_txt_file(data_dir: str, filenames: list, chunk_size: int = 500, overlap: int = 50):
    if vector_store._collection.count() > 0:
        print("Data already stored in ChromaDB. Skipping storage.")
        return

    all_documents = []
    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)

            markdown_content = content["markdown"]
            metadata = content["metadata"]

            chunks = split_text_into_chunks(markdown_content, chunk_size, overlap)
            # remove more then one space and more then one new line
            chunks = [re.sub(r'\s+', ' ', chunk) for chunk in chunks]
            for i, chunk in enumerate(chunks):
                metadata["chunk_index"] = i
                document = Document(page_content=chunk, metadata=metadata)
                all_documents.append(document)
    
    # Add all documents to the vector store
    # vector_store.add_documents(all_documents)
    print("Data successfully stored in ChromaDB.")
    print(f"Total documents to store: {len(all_documents)}")

# Main function
if __name__ == "__main__":
    data_dir = "src/input_data"
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    store_file_in_chromadb_txt_file(data_dir, filenames)