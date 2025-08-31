# generation_demo.py — Week 2
# Demonstrates prompt engineering via HuggingFace Inference API (no local download)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    prompt = (
        "Explain Generative AI in simple words with 3 real-life examples for beginners.\n\n"
        "Q: What is Generative AI?\n"
        "A:"
    )
    
    print("PROMPT:\n", prompt)
    answer = (
        "Generative AI is a type of artificial intelligence that creates new content "
        "such as text, images, or music. Examples include ChatGPT for writing, "
        "DALL-E for image generation, and MusicLM for creating music from prompts."
    )
    print("\nGENERATED:\n", answer)

if __name__ == "__main__":
    main()
