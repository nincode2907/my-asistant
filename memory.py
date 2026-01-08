import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os

# Initialize paths and models
DB_PATH = "./memory_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure DB directory exists
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Initialize ChromaDB Client
# PersistentClient saves data to disk
client = chromadb.PersistentClient(path=DB_PATH)
collection_name = "user_memory"
collection = client.get_or_create_collection(name=collection_name)

# Initialize Embedding Model
# We load it once here. Streamlit caching in app.py or backend logic might invoke this.
# Since backend.py imports this, it will load when backend loads.
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def add_memory(text):
    """
    Embeds and saves the text to the vector database.
    """
    embedding = model.encode(text).tolist()
    # Using UUID for unique ID
    mem_id = str(uuid.uuid4())
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[mem_id]
    )
    return f"Memory saved: '{text}'"

def get_relevant_context(query, n_results=3):
    """
    Retrieves the top N most relevant documents for the query.
    Returns a formatted string of context.
    """
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Check if we have results
    if not results['documents'] or not results['documents'][0]:
        return ""
        
    # Format results
    documents = results['documents'][0]
    # Filter out very distant matches if needed, but for now just return top N
    
    formatted_context = "\n".join([f"- {doc}" for doc in documents])
    return formatted_context

def delete_similar_memory(text_query, threshold=0.35):
    """
    Finds and deletes the most similar memory if it matches the threshold.
    Returns a status string.
    """
    query_embedding = model.encode(text_query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=['distances', 'documents']
    )
    
    # Check if we found anything
    if not results['ids'] or not results['ids'][0]:
        return "Không tìm thấy ký ức nào để xóa."
        
    distance = results['distances'][0][0]
    found_id = results['ids'][0][0]
    found_doc = results['documents'][0][0]
    
    if distance < threshold:
        collection.delete(ids=[found_id])
        return f"Đã quên thông tin cũ: '{found_doc}'"
    else:
        return f"Không tìm thấy ký ức đủ giống để xóa (độ sai biệt {distance:.2f} >= {threshold})."
