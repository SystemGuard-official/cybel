from src.create_knowledge_bank import store_file_in_chromadb_txt_file
import os

data_dir = "src/input_data"
filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]

store_file_in_chromadb_txt_file(data_dir, filenames, chunk_size=1000, overlap=50)