# File: data/retrieval/embed_and_index.py
# embed_and_index.py builds a FAISS index from text chunks using SentenceTransformer embeddings
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Correct relative paths
INPUT_FILE = "data/cleaned/aot_chunks.jsonl"
INDEX_FILE = "data/retrieval/aot_index.faiss"
METADATA_FILE = "data/retrieval/aot_metadata.json"

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def build_index(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index

if __name__ == "__main__":
    os.makedirs("data/retrieval", exist_ok=True)

    chunks = load_chunks(INPUT_FILE)
    index = build_index(chunks)

    faiss.write_index(index, INDEX_FILE)
    save_metadata(chunks, METADATA_FILE)

    print(f"✅ FAISS index saved to: {INDEX_FILE}")
    print(f"✅ Metadata saved to: {METADATA_FILE}")
    print(f"✅ Total entries indexed: {len(chunks)}")
