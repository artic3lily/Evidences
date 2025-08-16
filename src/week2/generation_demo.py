# generation_demo.py — Week 2
# Demonstrates prompt engineering via HuggingFace Inference API (no local download)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_utils import call_llm

def main():
    prompt = (
        "You are a math tutor. Help a student understand fractions.\n\n"
        "Q: What is 3/4 as a decimal?\n"
        "A:"
    )
    print("PROMPT:\n", prompt)
    answer = call_llm(prompt, max_new_tokens=80)
    print("\nGENERATED:\n", answer)

if __name__ == "__main__":
    main()
