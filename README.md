---
title: AOT AI
emoji: 👁
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: mit
short_description: 'Interactive multimodal AI app '
---


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
Anime Fanfiction Generator
This project is a local web app that creates anime-style fanfiction scenes (inspired by Attack on Titan) using a user's prompt and a local AI model.

Features
Takes user input to generate a dramatic anime scene

Uses a local embedding and retrieval system to add context

Outputs short, emotional, and intense fanfiction text

Runs completely locally, no internet needed

Files
app.py: Runs the web server that handles user requests

generate_local.py: Generates text using a local language model

retrieve.py: Finds related text chunks based on the user’s prompt

embed_and_index.py: Creates and stores vector embeddings for fast search

fanfic_prompt.txt: A prompt template for generating anime-style scenes

How to Run
Set up environment:
Make sure Python 3.8+ is installed. Install required packages:

bash
Copy
Edit
pip install flask sentence-transformers
Index your data (if needed):
If you want to add your own story data:

bash
Copy
Edit
python embed_and_index.py
Run the app:

bash
Copy
Edit
python app.py
Use it:
Go to http://localhost:5000 in your browser and enter a prompt!

Example Prompt
pgsql
Copy
Edit
Write a scene where Mikasa finds out Eren betrayed her.
Output Style
Short, dramatic lines

Inner thoughts

Cliffhanger endings

Anime pacing and tone

