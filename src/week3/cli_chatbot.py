# cli_chatbot.py — Week 3
# Simple CLI chatbot with in-memory conversation history using HF Inference API
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_utils import call_llm

SYSTEM_PROMPT = (
    "You are a helpful tutor who teaches math and Python to students. "
    "Be concise, clear and encouraging.\n\n"
)

def main():
    print("Chatbot Tutor — type 'exit' to quit.\n")
    history = ""

    while True:
        user_in = input("You: ").strip()
        if user_in.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        prompt = (
            SYSTEM_PROMPT
            + history
            + f"User: {user_in}\nTutor:"
        )
        answer = call_llm(prompt, max_new_tokens=150)
        print(f"Tutor: {answer}\n")

        # Keep a rolling window of last 3 turns to stay within token limits
        history += f"User: {user_in}\nTutor: {answer}\n"
        turns = history.strip().split("\n")
        if len(turns) > 12:          # ~6 dialogue turns
            history = "\n".join(turns[-12:]) + "\n"

if __name__ == "__main__":
    main()