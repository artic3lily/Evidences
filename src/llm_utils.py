"""
llm_utils.py — shared LLM helper for the Chatbot Tutor project.

Uses the HuggingFace Inference API so NO model is downloaded locally.
Set HF_TOKEN in a .env file or as an environment variable for higher rate limits.
Free tier works without a token for small requests.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Free, small, fast model on HF Inference API
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def get_headers():
    token = os.getenv("HF_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def call_llm(prompt: str, max_new_tokens: int = 150) -> str:
    """Call HuggingFace Inference API and return the generated text."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        }
    }
    try:
        resp = requests.post(HF_API_URL, headers=get_headers(), json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        return str(data)
    except requests.exceptions.HTTPError as e:
        # Model may be loading — return a friendly message
        return f"[Model is loading, please retry in ~20s. Error: {e}]"
    except Exception as e:
        return f"[LLM Error: {e}]"
