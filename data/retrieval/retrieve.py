# retrieve.py
## retrieve.py provides functions to retrieve relevant text chunks from a pre-built FAISS index
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "data/retrieval/aot_index.faiss"
METADATA_FILE = "data/retrieval/aot_metadata.json"
MODEL_NAME = 'all-MiniLM-L6-v2'

# 🔁 Load once (global cache)
_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    _metadata = json.load(f)

def embed_query(query):
    return np.array([_model.encode(query)])

def retrieve_relevant_chunks(query, k=2):
    query_embedding = embed_query(query)
    scores, indices = _index.search(query_embedding, k * 3)  # overfetch
    results = []
    for i in indices[0]:
        if i < len(_metadata) and "attack on titan" in _metadata[i].get("source", "").lower():
            results.append(_metadata[i])
        if len(results) >= k:
            break
    return "\n\n".join([f"[{r['title']}]\n{r['text']}" for r in results])


def retrieve_lore_context(query, k=3):
    print(f"🔍 Retrieving lore context for: {query}")
    query_embedding = embed_query(query)
    scores, indices = _index.search(query_embedding, k)
    results = [_metadata[i] for i in indices[0] if i < len(_metadata)]
    return "\n\n".join([f"[{r['title']}]\n{r['text']}" for r in results])

def search(query, k=3):
    print(f"🔍 Searching for: {query}")
    query_vector = embed_query(query)
    scores, indices = _index.search(query_vector, k)
    return [_metadata[i] for i in indices[0] if i < len(_metadata)]

if __name__ == "__main__":
    user_query = input("Ask a question about Attack on Titan: ")
    top_chunks = search(user_query, k=1)

    print("\n🔗 Top results:\n")
    for i, chunk in enumerate(top_chunks):
        print(f"{i+1}. [{chunk['source']}] {chunk['title']}\n{chunk['text']}\n")
