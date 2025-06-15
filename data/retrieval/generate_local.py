# generate_local.py creates a local model

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.retrieval.retrieve import retrieve_relevant_chunks
import json

CACHE_FILE = "output/answer_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        answer_cache = json.load(f)
else:
    answer_cache = {}

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# 🔁 Load model and tokenizer
print("🔁 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
torch.set_grad_enabled(False)

# ✅ Set pad token early
tokenizer.pad_token = tokenizer.eos_token

# ✅ Device setup
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Device set to use {device}")
model.to(device)

# 📝 Prompt templates
factual_template = """
You are an expert on Attack on Titan. Use the following context to answer the fan's question.

{context}

Question: {question}
Answer:"""

fanfic_template = """
You are an anime fanfiction co-pilot for Attack on Titan. Use the following story context and character tone to create a creative scene or dialogue.

Context:
{context}

Creative Prompt: {question}

Write in-character with anime-style narration and strong emotional tone.
Output:
"""

def load_prompt_template(filepath):
    with open(filepath, "r") as f:
        return f.read()

# ✂️ Trims context to stay within token limits
def trim_context_to_token_limit(context, question, template, tokenizer, max_tokens=1800):
    context = context.strip()
    while True:
        prompt = template.format(context=context, question=question)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"]
        if input_ids.shape[1] <= max_tokens:
            return prompt
        context = context[:int(len(context) * 0.9)]

# 🧠 Main answer generator
def generate_answer(question, fanfic_mode=False, max_tokens=150, temperature=0.8, top_p=0.95):

    t0 = time.time()
    context = retrieve_relevant_chunks(question, k=5)
    t1 = time.time()

    template_path = "data/prompts/fanfic_prompt.txt" if fanfic_mode else "data/prompts/factual_prompt.txt"
    template = load_prompt_template(template_path)
    prompt = trim_context_to_token_limit(context, question, template, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    t2 = time.time()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=fanfic_mode,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )


    t3 = time.time()

    print(f"⏱️ Retrieval: {t1 - t0:.2f}s | Tokenization: {t2 - t1:.2f}s | Generation: {t3 - t2:.2f}s")
    return tokenizer.decode(output[0], skip_special_tokens=True), prompt

# 🎭 Interactive loop
if __name__ == "__main__":
    print("🎭 Choose mode:")
    print("1. Lore Q&A")
    print("2. Fanfiction generation")
    mode = input("Enter mode (1 or 2): ").strip()
    fanfic_mode = (mode == "2")
    try:
        max_tokens = int(input("🔢 Max generation tokens (e.g., 100–250): ").strip())
    except:
        max_tokens = 150  # fallback default

    try:
        temperature = float(input("🔥 Sampling temperature (e.g., 0.7–1.0): ").strip())
    except:
        temperature = 0.8

    try:
        top_p = float(input("🎯 Top-p sampling (e.g., 0.8–1.0): ").strip())
    except:
        top_p = 0.95

    log_file = "output/fanfic_log.txt" if fanfic_mode else "output/factual_log.txt"
    os.makedirs("output", exist_ok=True)

    while True:
        question = input("🗣️ Ask a question about AoT (or press Enter to regenerate last): ").strip()
        if not question:
            if 'last_question' not in globals():
                print("⚠️ No previous question to regenerate.")
                continue
            question = last_question
        else:
                last_question = question
        print(f"🔍 Searching for: {question}")
        if question in answer_cache:
            print("🔁 Retrieved from cache")
            answer = answer_cache[question]
            full_prompt = "(cached)"
        else:
            answer, full_prompt = generate_answer(question, fanfic_mode=fanfic_mode, temperature=temperature, max_tokens=max_tokens)
            answer_cache[question] = answer

        print("\n🧠 Fanfiction Co-Pilot says:\n" if fanfic_mode else "\n📚 Lore Master says:\n")
        print(answer)
        print("\n" + "-" * 50 + "\n")

        with open(log_file, "a") as f:
            f.write(f"Full Prompt:\n{full_prompt}\n\n")
            f.write(f"Prompt: {question}\n\n")
            f.write(f"Generated Answer:\n{answer}\n")
            f.write("=" * 80 + "\n\n")
        print(f"✅ Answer logged to {log_file}\n")
        
        answer_cache[question] = answer
        with open(CACHE_FILE, "w") as f:
            json.dump(answer_cache, f, indent=2)
