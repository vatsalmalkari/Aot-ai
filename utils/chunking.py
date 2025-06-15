## utils/chunking.py
# This module provides functions to chunk text into smaller pieces based on a maximum token limit.
import textwrap

def chunk_text(text, max_tokens=500):
    sentences = text.split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        if len((current + sentence).split()) <= max_tokens:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "

    if current:
        chunks.append(current.strip())

    return chunks
