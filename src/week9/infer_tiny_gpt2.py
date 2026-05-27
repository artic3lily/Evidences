import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "Explain RAG to a beginner.",
    "Why is fine-tuning useful?",
    "What is prompt engineering?",
]


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    formatted_prompt = f"Prompt: {prompt}\nAnswer:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in decoded:
        return decoded.split("Answer:", 1)[1].strip()
    return decoded.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    print(f"Loaded model: {args.model_path}")
    print()

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens)
        print(f"Output: {answer}")
        print("-" * 80)


if __name__ == "__main__":
    main()
