# LO9 Model Development Evidence

This mini-project adds a small, realistic fine-tuning workflow that you can use as evidence for LO9.

## Objective

Develop and fine-tune a small generative AI model for beginner-friendly tutor responses.

## Recommended Base Model

- `sshleifer/tiny-gpt2` for weaker laptops
- `distilgpt2` if you want a slightly stronger base model

The scaffold below defaults to `sshleifer/tiny-gpt2` because it is much lighter and faster to train.

## Files Added For LO9

- `src/week9/data/tutor_qa.jsonl`
- `src/week9/train_tiny_gpt2.py`
- `src/week9/infer_tiny_gpt2.py`
- `notebooks/lo9_finetune.ipynb`

## Screenshot Checklist

Take these screenshots after running the notebook or scripts:

1. Dataset sample
2. Model loading and tokenizer setup
3. Hyperparameters
4. Training logs with loss output
5. Before fine-tuning inference
6. After fine-tuning inference
7. Repo structure showing the LO9 files
8. Optional: saved model folder in `artifacts/week9_lo9`

## Suggested Commands

Run a quick before snapshot:

```powershell
python src/week9/infer_tiny_gpt2.py --model-path sshleifer/tiny-gpt2
```

Train the small model:

```powershell
python src/week9/train_tiny_gpt2.py --epochs 2
```

Run the after snapshot:

```powershell
python src/week9/infer_tiny_gpt2.py --model-path artifacts/week9_lo9/tiny-gpt2-tutor
```

## Report Template

### Objective

To develop and fine-tune a small generative AI model for beginner-friendly tutor responses.

### Base Model

`sshleifer/tiny-gpt2`

### Why This Model Was Chosen

A small model was selected because it is faster to train, requires fewer computational resources, and is suitable for demonstrating fine-tuning in a limited environment.

### Dataset Preparation

A custom dataset of prompt-response pairs was created for educational Q&A. The data focuses on simple explanations of AI and programming topics.

### Fine-Tuning Method

The project uses Hugging Face Transformers for lightweight causal language model fine-tuning. The training script also supports optional LoRA if `peft` is installed, but the default workflow remains simple enough for limited hardware.

### Challenges Faced

- Training speed on limited hardware
- Keeping the dataset small but useful
- Avoiding overfitting on repeated question styles
- Choosing a model small enough to run locally

### Result

The fine-tuned model should produce more tutor-like and structured responses than the untuned base model for the custom prompts in the dataset.
