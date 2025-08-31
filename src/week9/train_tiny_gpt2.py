import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


DEFAULT_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "tutor_qa.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "artifacts" / "week9_lo9" / "tiny-gpt2-tutor"


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example(example: dict) -> str:
    return f"Prompt: {example['prompt']}\nAnswer: {example['response']}"


class TutorQADataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.records = []
        for example in examples:
            text = format_example(example) + tokenizer.eos_token
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            self.records.append({key: torch.tensor(value) for key, value in tokenized.items()})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def maybe_apply_lora(model, use_lora: bool):
    if not use_lora:
        return model

    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "LoRA requested but 'peft' is not installed. Install it with: pip install peft"
        )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn"],
    )
    return get_peft_model(model, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--use-lora", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(dataset_path)
    print(f"Loaded {len(examples)} examples from {dataset_path}")
    print("Sample training example:")
    print(format_example(examples[0]))
    print()
    print("Hyperparameters:")
    print(f"  model_name     = {args.model_name}")
    print(f"  epochs         = {args.epochs}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  learning_rate  = {args.learning_rate}")
    print(f"  max_length     = {args.max_length}")
    print(f"  use_lora       = {args.use_lora}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = maybe_apply_lora(model, args.use_lora)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = TutorQADataset(examples, tokenizer, args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.model_name,
        "dataset": str(dataset_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "use_lora": args.use_lora,
        "num_examples": len(examples),
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print(f"Training finished. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
