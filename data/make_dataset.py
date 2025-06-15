## make_dataset.py
# This script processes episode summaries and character information from JSONL files,
# chunks them into smaller pieces, and saves the results in a new JSONL file.
import json
import os
from utils.chunking import chunk_text

EPISODE_FILE = "data/cleaned/aot_episode_chunks.jsonl"
CHARACTER_FILE = "data/cleaned/aot_character_chunks.jsonl"
OUTPUT_FILE = "data/cleaned/aot_chunks.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(records)} chunks to {path}")

def main():
    episodes = load_jsonl(EPISODE_FILE)
    characters = load_jsonl(CHARACTER_FILE)

    combined = episodes + characters
    all_chunks = []

    for doc in combined:
        chunks = chunk_text(doc["text"], max_tokens=500)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "title": doc["title"],
                "chunk_id": i,
                "text": chunk
            })

    save_jsonl(all_chunks, OUTPUT_FILE)

if __name__ == "__main__":
    main()
