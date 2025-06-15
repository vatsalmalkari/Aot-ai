## This script processes episode summaries from text files and saves them in a structured JSONL format.
import os
import json

RAW_FOLDER = "data/raw/episodes"
OUT_FILE = "data/cleaned/aot_episode_chunks.jsonl"

def load_episode_summaries(folder):
    all_docs = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()

        if len(lines) < 2 or not lines[0].startswith("Title:") or not lines[1].startswith("Summary:"):
            print(f"⚠️ Skipped (unexpected format): {filename}")
            continue

        title = lines[0].replace("Title:", "").strip()
        summary = lines[1].replace("Summary:", "").strip()

        all_docs.append({
            "source": "episode_summary",
            "title": title,
            "text": summary
        })

    return all_docs


def save_to_jsonl(docs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(docs)} episode summaries to {output_path}")


if __name__ == "__main__":
    episodes = load_episode_summaries(RAW_FOLDER)
    save_to_jsonl(episodes, OUT_FILE)
