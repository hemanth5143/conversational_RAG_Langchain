import json
from sentence_transformers import SentenceTransformer
import os
import chromadb
from chromadb.utils import embedding_functions

with open(r"C:\Users\hsai5\OneDrive\Documents\LLM projects\conversational_RAG_chatbot\data\Alexander_Street_shareGPT_2.0.json", 'r') as file:
    dataset = json.load(file)
print("Dataset loaded successfully.")

directory = r"C:\Users\hsai5\OneDrive\Documents\LLM projects\conversational_RAG_chatbot\chroma_db"
chroma_client = chromadb.PersistentClient(path=directory)

print("ChromaDB client initialized.")

embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

class CustomEmbeddingFunction:
    def __call__(self, input: str) -> list:
        embedding = embedding_model.encode(input)
        return embedding.tolist()

# Create an instance of the custom embedding function
embedding_func = CustomEmbeddingFunction()
print("Embedding function defined.")

# Create a collection in ChromaDB
collection = chroma_client.create_collection(
    name="mindguardian_collection",
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)
print("ChromaDB collection created.")

from more_itertools import chunked

import uuid

def store_embedded_data_in_chromadb(dataset, batch_size=10):
    global_id = 0  # Start a global counter
    for batch in chunked(dataset, batch_size):
        ids = []
        documents = []
        metadatas = []

        for data in batch:
            merged_text = data['input'] + " " + data['output']
            ids.append(str(global_id))  # Use the global counter for unique IDs
            documents.append(merged_text)
            metadatas.append(data)  # Store original data as metadata
            global_id += 1  # Increment the global counter

        # Add data to the ChromaDB collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    print("Data stored in ChromaDB.")

store_embedded_data_in_chromadb(dataset)  