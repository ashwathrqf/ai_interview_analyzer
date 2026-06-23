"""Ollama client wrapper. Talks to a local Ollama server — no API key, no cost.

Requires Ollama running locally (https://ollama.com) with a model pulled, e.g.:
    ollama pull llama3.1
"""

import ollama

# Change this if you pull a different/smaller model (e.g. "llama3.2:3b", "phi3")
MODEL = "llama3.1"


def get_client():
    """Returns the ollama module itself — kept as a function so the rest of the
    codebase doesn't care whether we're using Ollama, OpenAI, or Anthropic under
    the hood. Only this file would need to change to swap providers later.
    """
    return ollama


def chat(system_prompt: str, user_prompt: str) -> str:
    """Send a system + user prompt to the local model and return the text reply."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]