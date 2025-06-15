## File: data/retrieval/generate.py
# generate.py uses OpenAI's API to answer questions about Attack on Titan using retrieved context chunks
import openai
from openai import OpenAI
from retrieve import search
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(query, context_chunks):
    context = "\n\n".join([f"[{c['title']}]\n{c['text']}" for c in context_chunks])
    return (
        f"You are an expert on Attack on Titan. Use the following context to answer the fan's question.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

def generate_answer(prompt, model="gpt-3.5-turbo", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = input("🗣️ Ask a question about AoT: ")
    top_chunks = search(query, k=5)
    prompt = build_prompt(query, top_chunks)
    answer = generate_answer(prompt)

    print("\n🧠 Fanfiction Co-Pilot says:\n")
    print(answer)

