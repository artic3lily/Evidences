# conversational_rag_cli.py - Week 5

HARMFUL_KEYWORDS = [
    "hack",
    "password",
    "steal",
    "bypass",
    "attack",
    "exploit",
]

def main():
    print("Conversational RAG Tutor - type 'exit' to quit.\n")
    history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        lowered_question = question.lower()

        if any(keyword in lowered_question for keyword in HARMFUL_KEYWORDS):
            answer = (
                "I can't help with hacking, privacy invasion, or harmful activity. "
                "If you need account access, please use official recovery methods."
            )
        elif "explain it simply" in lowered_question:
            answer = (
                "In simple words, RAG first finds useful information and then "
                "uses it to generate a better answer."
            )
        elif "rag" in lowered_question:
            answer = (
                "RAG stands for Retrieval-Augmented Generation. "
                "It combines document retrieval with a language model."
            )
        else:
            answer = (
                "Generative AI creates new content such as text, images, and music."
            )

        print(f"Tutor: {answer}\n")
        history.append((question, answer))

if __name__ == "__main__":
    main()
