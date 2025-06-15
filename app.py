# app.py
# This file runs a web server that takes a user prompt, finds related info, and creates an anime-style fanfiction scene using a local AI model.
# This app uses Gradio for the web interface and PyTorch for the AI model.
import os
from datetime import datetime
import time
import torch
import gradio as gr
from PIL import Image
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig
)
from data.retrieval.retrieve import retrieve_relevant_chunks 

# Ensure nltk punkt tokenizer is downloaded only once
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# --- Configuration and Model Loading ---
LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Prioritize MPS for Apple Silicon, then CUDA for NVIDIA GPUs, fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
BLIP_NAME = "Salesforce/blip-image-captioning-large"

# Load image captioning model
caption_processor = BlipProcessor.from_pretrained(BLIP_NAME)
caption_model = BlipForConditionalGeneration.from_pretrained(BLIP_NAME).to(device)

# Optional quantization for CUDA only
bnb_config = None
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

print(f"💻 Device set to use {device}")

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    quantization_config=bnb_config if device == "cuda" else None,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
)
tokenizer.pad_token = tokenizer.eos_token
model.eval() # Set model to evaluation mode
torch.set_grad_enabled(False) # Disable gradient computation for inference
if device != "cuda": # Move to device for non-CUDA (CPU/MPS)
    model.to(device)

# Try compiling models for speed (only works with PyTorch 2.0+)
try:
    model = torch.compile(model)
    print("✅ LLM compiled for faster inference")
except Exception as e:
    print(f"⚠️ Skipped LLM compile for LLM: {e}")
try:
    caption_model = torch.compile(caption_model)
    print("✅ BLIP compiled for faster inference")
except Exception as e:
    print(f"⚠️ Skipped BLIP compile: {e}")

# --- Core Logic Functions ---
def get_prompt_template(mode, style=""):
    """Returns the appropriate prompt template based on the selected mode and style."""
    if mode == "Fanfiction":
        return (
            "You are an anime fanfiction writer for Attack on Titan.\n\n"
            "Context:\n{context}\n\n" # Context will be image caption if provided
            "Creative Prompt: {question}\n\n"
            "Write a dramatic scene of betrayal and confrontation. Use short, emotional lines, heavy internal thoughts, and anime-style pacing. End on a tense moment or cliffhanger."
        )
    elif mode == "Lore Q&A":
        tail = ""
        if style == "Concise":
            tail = "Answer in 1–2 clear factual sentences, strictly adhering to the context provided. Do not invent information."
        elif style == "Narrative":
            tail = "Answer with rich context, emotional tone, and detailed explanations, drawing from the provided context. Maintain a narrative flow."
        elif style == "Quote-heavy":
            tail = "Include relevant quotes from the Attack on Titan characters within your detailed answer, ensuring they fit the context."
        return (
            "You are a lore expert on Attack on Titan. Only answer with detailed character backstory, personality traits, or history within the AoT universe, based *only* on the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\nAnswer:\n" + tail
        )
    else: # Summarization
        return "You are a summarization assistant. Summarize the following input concisely and accurately:\n\n{context}\n\nSummary:"

def generate_caption(image: Image.Image) -> str:
    """Generates a descriptive caption for the given PIL Image."""
    if image is None:
        return ""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = caption_processor(image, return_tensors="pt").to(device)
    # Using num_beams for potentially better quality captions
    out = caption_model.generate(**inputs, max_new_tokens=30, num_beams=4)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def trim_context(context_str, question_str, template_str, current_tokenizer, max_tokens):
    """
    Trims the context to ensure the total prompt length (context + question + template)
    does not exceed the model's maximum input tokens.
    """
    # Calculate tokens taken by template and question without context
    dummy_prompt = template_str.format(context="", question=question_str)
    question_template_tokens = current_tokenizer(dummy_prompt, return_tensors="pt")["input_ids"].shape[1]

    # Leave some buffer tokens (e.g., 10) for minor variations/special tokens
    available_tokens_for_context = max_tokens - question_template_tokens - 10

    if not context_str.strip():
        return template_str.format(context="", question=question_str)

    # Tokenize and truncate context
    context_ids = current_tokenizer(context_str, max_length=available_tokens_for_context, truncation=True, return_tensors="pt")["input_ids"]
    trimmed_context = current_tokenizer.decode(context_ids[0], skip_special_tokens=True)

    return template_str.format(context=trimmed_context, question=question_str)

def generate_answer_web(mode, input_text, image, max_tokens, temperature, top_p, style):
    """
    Main function to generate AI answers based on mode, input, and settings.
    Returns context, image caption, and the generated answer.
    """
    start_time = time.time()
    caption_text = ""
    context = ""
    question_for_llm = input_text # Default question for LLM

    # Step 1: Generate Image Caption if image is provided
    if image:
        try:
            caption_text = generate_caption(image)
        except Exception as e:
            caption_text = f"Error generating caption: {e}"
            print(f"Error generating caption: {e}")

    # Step 2: Prepare Context based on Mode
    if mode == "Summarization":
        # For summarization, context is either the caption or the input text
        context = caption_text if caption_text else input_text
        question_for_llm = "" # No specific question for summarization
    elif mode == "Fanfiction":
        # For fanfiction, context is primarily the image caption
        context = caption_text
        # Input text becomes the "creative prompt"
    else: # Lore Q&A
        # Retrieve text chunks from database
        retrieved_text = retrieve_relevant_chunks(input_text, k=5)
        # Combine image caption with retrieved text for Lore Q&A context
        if caption_text:
            context = f"[Image Caption]: {caption_text}\n\n" + retrieved_text
        else:
            context = retrieved_text

    # Step 3: Get Prompt Template and Format Prompt
    template_str = get_prompt_template(mode, style)
    formatted_prompt = trim_context(context, question_for_llm, template_str, tokenizer, max_tokens)

    # Step 4: Generate LLM Output
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Sampling strategy based on mode/style
    use_sampling = (mode == "Fanfiction" or style == "Quote-heavy")
    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=use_sampling, # Enable sampling for creative tasks
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1 # Helps prevent repetitive phrases
        )
        decoded_answer = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    except Exception as e:
        decoded_answer = f"Error during generation: {e}"
        print(f"Error during generation: {e}")

    # Step 5: Log output for debugging/record-keeping
    os.makedirs("output", exist_ok=True)
    log_filename = f"output/{mode.lower().replace(' ', '_')}_log.txt"
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()}\nMode: {mode} | Style: {style}\n")
        f.write(f"User Input: {input_text}\n")
        f.write(f"Image Caption: {caption_text}\n\n")
        f.write(f"Context used by LLM:\n{context}\n\n")
        f.write(f"Full Prompt Sent to LLM:\n{formatted_prompt}\n\n")
        f.write(f"Generated Output:\n{decoded_answer}\n")
        f.write("=" * 80 + "\n\n")
    print(f"Output logged to {log_filename}")

    return context, caption_text, decoded_answer

# === Gradio UI Construction ===

# Helper functions for page visibility
def show_main_app():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def show_help_page():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def show_welcome_page():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

# Define components and layouts for each "page" within their own Blocks context
# This helps ensure components are registered correctly.

# Welcome Page
with gr.Blocks(title="Welcome to AoT AI!") as welcome_page_content:
    gr.Markdown(
        """
        # 🛡️ Welcome to Attack on Titan AI! 🚀
        This website is your ultimate friend for exploring the world of Attack on Titan and beyond.
        It combines advanced AI models to bring you a unique interactive experience:

        ### What it is:
        * **Attack on Titan Lore Expert:** Dive deep into the rich history, characters, and events of AoT. Ask specific questions and get detailed, context-aware answers.
        * **Creative Fanfiction Generator:** Unleash your imagination! Provide a prompt (and optionally an image) to generate dramatic and engaging Attack on Titan fanfiction scenes in an anime style.
        * **General Summarization & Image Captioning:** Need to condense a long piece of text or get a quick overview of an image? Our AI can summarize text inputs and generate descriptive captions for your uploaded images.

        ### How It Works:
        The system uses a powerful **Large Language Model (LLM)** for text generation and a specialized **Image Captioning Model (BLIP)** to understand your images. For Attack on Titan lore, it uses a **Retrieval-Augmented Generation (RAG)** system that pulls relevant information from a vast database of AoT knowledge to provide accurate and detailed answers.

        ### Get Started:
        Click "Start" to begin your journey into the world of AoT AI!
        """
    )
    with gr.Row():
        start_button_welcome = gr.Button("🚀 Start")
        help_button_welcome = gr.Button("❓ Learn More & Help")

# Main Application Page
with gr.Blocks(title="AoT Q&A, Fanfic, Summarizer + Image Captioning") as main_app_content:
    gr.Markdown("## 🛡️ Attack on Titan AI Interface")

    # Dynamic warning for summarization mode
    summarization_warning = gr.Markdown(
        "⚠️ **You are in Summarization mode.** This is for summarizing text or captioning images. "
        "Use **Lore Q&A** for specific questions about Attack on Titan characters or story.",
        visible=False,
        label="Mode Information" # Added label for clarity
    )

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["Lore Q&A", "Fanfiction", "Summarization"],
                value="Lore Q&A",
                label="🎮 Select Mode",
                info="Choose how you want the AI to respond."
            )
        with gr.Column(scale=1):
            # Answer style is only relevant for Lore Q&A, default to Narrative
            style = gr.Radio(
                choices=["Concise", "Narrative", "Quote-heavy"],
                value="Narrative",
                label="📝 Answer Style (for Lore Q&A)",
                info="How the AI should format its answer for Lore Q&A.",
                visible=True
            )

    # Input section with clear labels and placeholders
    with gr.Row():
        input_text = gr.Textbox(
            label="📝 Your Text Input / Question",
            lines=3,
            placeholder="Type your Attack on Titan question, creative fanfiction prompt, or text to summarize here...",
            info="This is where you'll provide the text for the AI to process."
        )
        image = gr.Image(
            type="pil",
            label="🖼 Upload Image (Optional)"
        )

    # Generation settings accordion with improved descriptions
    with gr.Accordion("⚙️ Generation Settings (Advanced)", open=True):
        gr.Markdown(
            """
            These settings control how the AI generates its output. Experiment with them to get different results!
            * **🔢 Max New Tokens to Generate:** This sets the maximum length of the AI's response. A higher number means a potentially longer answer or fanfiction scene.
            * **🔥 Temperature:** This controls the creativity and randomness of the AI's output.
                * **Lower (e.g., 0.1-0.5):** More factual, predictable, and less creative. Good for precise answers.
                * **Higher (e.g., 0.8-1.5):** More diverse, surprising, and creative. Often used for imaginative writing like fanfiction.
            * **🎯 Top-p:** This refines the diversity of the output. It helps ensure quality while still allowing for creativity.
                * **Lower (e.g., 0.1-0.5):** More focused and predictable text.
                * **Higher (e.g., 0.9-1.0):** More diverse and varied responses. A good balance is often around 0.95.
            """
        )
        max_tokens = gr.Slider(50, 500, value=200, step=10, label="🔢 Max New Tokens to Generate", info="Controls the length of the AI's output.")
        temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="🔥 Temperature", info="Adjust for creativity (higher = more creative, lower = more factual).")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="🎯 Top-p", info="Filters less probable words for better quality diversity.")
        generate_button = gr.Button("🚀 Generate Response", variant="primary") # Highlight primary action

    # Output section with clear labels and more space
    with gr.Row():
        context_out = gr.Textbox(label="📚 Context Used by AI (Retrieved Info + Caption)", lines=8, interactive=False, max_lines=15)
        caption_out = gr.Textbox(label="🖼 Image Caption Generated", lines=2, interactive=False)
    answer_out = gr.Textbox(label="🧠 Generated Output", lines=10, interactive=False, max_lines=20)

    # Dynamic UI updates for mode selection
    def update_ui_visibility(selected_mode):
        if selected_mode == "Summarization":
            # Hide style options, show summarization warning
            return gr.update(visible=False), gr.update(visible=True)
        else:
            # Show style options, hide summarization warning
            return gr.update(visible=True), gr.update(visible=False)

    mode.change(
        fn=update_ui_visibility,
        inputs=mode,
        outputs=[style, summarization_warning],
        queue=False
    )

    # Safe generate function wrapper
    def safe_generate(mode_val, input_val, img, max_tok, temp, topp, style_val):
        # Pass an empty string for style if mode is summarization, as it's not applicable
        style_clean = "" if mode_val == "Summarization" else style_val
        return generate_answer_web(mode_val, input_val, img, max_tok, temp, topp, style_clean)

    generate_button.click(
        fn=safe_generate,
        inputs=[mode, input_text, image, max_tokens, temperature, top_p, style],
        outputs=[context_out, caption_out, answer_out],
        api_name="generate_output" # Give the API endpoint a name
    )

# Explanation Page
with gr.Blocks(title="Help & Explanations") as help_page_content:
    gr.Markdown(
        """
        # ❓ Help & Explanations

        ## How to Use This App:

        1.  **Choose a Mode:**
            * **Lore Q&A:** Ask questions about Attack on Titan characters, history, or plot.
            * **Fanfiction:** Get the AI to write a creative scene based on your prompt (and an optional image).
            * **Summarization:** Condense any text you provide, or get a caption for an image.

        2.  **Provide Input:** Type your question, fanfiction prompt, or text to summarize in the "Your Text Input / Question" box.

        3.  **Upload Image (Optional):** If you want an image caption or want your fanfiction to be inspired by an image, upload one.

        4.  **Adjust Settings (Optional, Advanced):** Modify "Generation Settings" if you want to control the AI's output length, creativity, or diversity.

        5.  **Click "Generate Response":** The AI will process your input and provide its output in the "Generated Output" box.

        ---

        ### 🎮 Mode Selection Explained:

        * **Lore Q&A:** This mode turns the AI into an expert on the Attack on Titan universe. Ask specific questions about characters (e.g., "Who is Eren Yeager?"), plot points, locations, or historical events within the AoT world. The AI uses a knowledge base to provide accurate and detailed answers.
        * **Fanfiction:** Unleash your creativity! In this mode, the AI acts as an Attack on Titan fanfiction writer. Provide a creative prompt (e.g., "Write a tense conversation between Levi and Zeke about their past"). You can also upload an image to inspire the scene. The AI aims to generate dramatic and engaging fanfiction with an anime-style flair.
        * **Summarization:** This is a general-purpose text and image processing mode.
            * **Text Summarization:** Paste any lengthy text, and the AI will condense it into a concise summary.
            * **Image Captioning:** Upload an image, and the AI will generate a descriptive caption for it, explaining what's visible in the picture.

        ---

        ### 📝 Answer Style (for Lore Q&A only):

        These options allow you to tailor the AI's response style specifically when you are using the "Lore Q&A" mode:

        * **Concise:** Get straight to the point! The AI will provide short, factual answers, typically in 1-2 clear sentences. Ideal for quick information retrieval.
        * **Narrative:** For a more immersive and detailed experience. The AI will answer with more context, emotional tone, and elaborate explanations, weaving a richer story around the information. This is the default setting.
        * **Quote-heavy:** If you love direct dialogue from the series! The AI will include relevant quotes from Attack on Titan characters within its detailed answers, making the information feel more authentic to the series.

        ---

        ### ⚙️ Generation Settings (Advanced Control):

        These sliders give you fine-tuned control over how the AI generates its text. Experiment with them to achieve different types of outputs!

        * **🔢 Max New Tokens to Generate:**
            * **What it does:** This sets the *maximum length* of the AI's generated response. A "token" is roughly equivalent to a word or part of a word.
            * **Impact:** A higher number allows for longer, more detailed answers or more extensive fanfiction scenes. A lower number ensures brief responses.
            * **Recommended:** Start with default (200) and increase if you need more content.

        * **🔥 Temperature:**
            * **What it does:** This is the primary control for the AI's *creativity* and *randomness*. It influences the probability of the AI choosing less common words.
            * **Impact:**
                * **Lower Temperature (e.g., 0.1 - 0.5):** The AI will stick to the most probable and "safe" words. This results in very factual, predictable, and less imaginative outputs. **Use for precise Lore Q&A.**
                * **Higher Temperature (e.g., 0.8 - 1.5):** The AI becomes more "adventurous," considering a wider range of less probable words. This leads to more diverse, surprising, and sometimes even unique outputs. **Ideal for creative Fanfiction.**

        * **🎯 Top-p:**
            * **What it does:** This is another advanced sampling parameter that controls the *diversity* of the output, working in conjunction with temperature. It tells the AI to only consider the most probable words that add up to a certain cumulative probability.
            * **Impact:**
                * **Lower Top-p (e.g., 0.1 - 0.5):** The AI considers a smaller, more focused set of highly probable words for its next token. This results in more coherent, less varied, and typically more factual text.
                * **Higher Top-p (e.g., 0.9 - 1.0):** The AI considers a broader range of possible words, including some less probable ones. This leads to more diverse and varied responses. Generally, a value around 0.9 or 0.95 is a good balance for creative tasks.

        ---

        ### Output Fields Explained:

        * **📚 Context Used by AI (Retrieved Info + Caption):** This textbox shows you the raw information that the AI used as background for generating its answer. For "Lore Q&A," this will include chunks of relevant Attack on Titan data retrieved from the knowledge base. If an image was uploaded, its generated caption will also be part of this context.
        * **🖼 Image Caption Generated:** If you upload an image, the AI's automatically generated textual description of that image will appear here. This is a direct output from the BLIP image captioning model.
        * **🧠 Generated Output:** This is the main result! This is where the AI's final answer, your fanfiction scene, or the summarized text will appear after processing your request.

        """
    )
    back_to_start_button = gr.Button("⬅️ Back to Start Page")


# --- Main Gradio App Layout (Containers for pages) ---
with gr.Blocks(title="Attack on Titan AI - Interactive Assistant") as app:
    # Containers for each "page"
    # Initially, only the welcome page is visible
    with gr.Column(visible=True, elem_id="welcome_col") as welcome_column:
        welcome_page_content.render() # Render the welcome page content inside this column

    with gr.Column(visible=False, elem_id="main_app_col") as main_app_column:
        main_app_content.render() # Render the main app content inside this column

    with gr.Column(visible=False, elem_id="help_col") as help_column:
        help_page_content.render() # Render the help page content inside this column

    # --- Button Click Event Bindings ---
    # Connect buttons from welcome_page_content to show/hide functions
    start_button_welcome.click(
        fn=show_main_app,
        outputs=[welcome_column, main_app_column, help_column]
    )

    help_button_welcome.click(
        fn=show_help_page,
        outputs=[welcome_column, main_app_column, help_column]
    )

    # Connect button from help_page_content to show/hide function
    back_to_start_button.click(
        fn=show_welcome_page,
        outputs=[welcome_column, main_app_column, help_column]
    )

# === Patch for Gradio Schema Bug ===
# This patch ensures compatibility with certain Gradio versions/environments
import gradio_client.utils as gc_utils
from gradio_client.utils import APIInfoParseError

def patched_json_schema_to_python_type(schema, defs=None):
    if schema is True or schema is False:
        return str
    return gc_utils._json_schema_to_python_type_original(schema, defs)

if not hasattr(gc_utils, "_json_schema_to_python_type_original"):
    gc_utils._json_schema_to_python_type_original = gc_utils._json_schema_to_python_type
    gc_utils._json_schema_to_python_type = patched_json_schema_to_python_type

# --- Launch the Gradio App ---
if __name__ == "__main__":
    app.launch(share=True) # share=True generates a public link