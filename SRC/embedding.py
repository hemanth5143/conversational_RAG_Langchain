import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Load dataset
with open(r"C:\Users\hsai5\OneDrive\Documents\LLM projects\conversational_RAG_chatbot\data\Alexander_Street_shareGPT_2.0.json", 'r') as file:
    dataset = json.load(file)
print("Dataset loaded successfully.")

# Initialize ChromaDB client
directory = "C:/Users/hsai5/OneDrive/Documents/LLM projects/conversational_RAG_chatbot/chroma_db"
chroma_client = chromadb.PersistentClient(path=directory)
print("ChromaDB client initialized.")

# Initialize embedding model
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

class CustomEmbeddingFunction:
    def __call__(self, input: str) -> list:
        embedding = embedding_model.encode(input)
        return embedding.tolist()

# Create an instance of the custom embedding function
embedding_func = CustomEmbeddingFunction()
print("Embedding function defined.")

def get_or_create_collection():
    collection_name = "mindguardian_collection"
    try:
        # Try to get the existing collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        print(f"Using existing ChromaDB collection: {collection_name}")
    except ValueError:
        # If the collection doesn't exist, create a new one
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new ChromaDB collection: {collection_name}")
    return collection

def store_embedded_data_in_chromadb(dataset, collection, batch_size=10):
    from more_itertools import chunked
    import uuid

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

if __name__ == "__main__":
    collection = get_or_create_collection()
    
    # Check if the collection is empty before adding data
    if collection.count() == 0:
        store_embedded_data_in_chromadb(dataset, collection)
    else:
        print("Collection already contains data. Skipping data storage.")