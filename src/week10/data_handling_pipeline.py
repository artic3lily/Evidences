import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = Path(__file__).resolve().parent / "data" / "raw_tutor_qa.csv"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "week10_lo10"
DEFAULT_TOKENIZER = (
    ROOT_DIR / "artifacts" / "week9_lo9" / "tiny-gpt2-tutor"
    if (ROOT_DIR / "artifacts" / "week9_lo9" / "tiny-gpt2-tutor").exists()
    else "sshleifer/tiny-gpt2"
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def print_heading(title: str):
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_heading("LO10 DATA HANDLING PIPELINE")
    print(f"Raw data path: {RAW_DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    print_heading("STEP 1: LOAD RAW DATA")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Original shape: {df.shape}")
    print("Raw sample rows:")
    print(df.head(5).to_string(index=False))

    print_heading("STEP 2: CLEAN TEXT")
    df["original_text"] = df["text"]
    missing_before = int(df["text"].isna().sum())
    df["cleaned_text"] = df["text"].apply(clean_text)

    before_examples = (
        df.loc[df["original_text"].notna(), ["original_text", "cleaned_text"]]
        .head(4)
        .copy()
    )

    df = df[df["cleaned_text"] != ""].copy()
    empty_removed = int((df["cleaned_text"].str.len() == 0).sum())
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["cleaned_text"]).copy()
    duplicates_removed = before_dedup - len(df)
    before_short_filter = len(df)
    df = df[df["cleaned_text"].str.len() > 20].copy()
    short_removed = before_short_filter - len(df)

    print(f"Missing rows before cleaning: {missing_before}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Short rows removed: {short_removed}")
    print(f"Rows remaining after cleaning: {len(df)}")
    print("\nBefore vs after cleaning examples:")
    for _, row in before_examples.iterrows():
        print(f"- BEFORE: {row['original_text']}")
        print(f"  AFTER : {row['cleaned_text']}")

    df["char_length"] = df["cleaned_text"].str.len()
    df["word_count"] = df["cleaned_text"].str.split().str.len()

    histogram_path = OUTPUT_DIR / "text_length_histogram.png"
    plt.figure(figsize=(8, 4))
    plt.hist(df["char_length"], bins=6, edgecolor="black")
    plt.title("Cleaned Text Length Distribution")
    plt.xlabel("Character Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()
    print(f"\nSaved histogram: {histogram_path}")

    cleaned_path = OUTPUT_DIR / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset: {cleaned_path}")

    print_heading("STEP 3: TRAIN / VALIDATION SPLIT")
    train_df = df.sample(frac=0.8, random_state=42).sort_values("id")
    val_df = df.drop(train_df.index).sort_values("id")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    train_path = OUTPUT_DIR / "train_dataset.csv"
    val_path = OUTPUT_DIR / "validation_dataset.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"Saved train split: {train_path}")
    print(f"Saved validation split: {val_path}")

    print_heading("STEP 4: TOKENIZATION PIPELINE")
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_TOKENIZER))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for frame in (train_df, val_df):
        tokenized = tokenizer(
            frame["cleaned_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=64,
        )
        frame.loc[:, "token_count"] = [
            int(sum(token != tokenizer.pad_token_id for token in ids))
            for ids in tokenized["input_ids"]
        ]

    train_token_path = OUTPUT_DIR / "train_tokenized_preview.json"
    val_token_path = OUTPUT_DIR / "validation_tokenized_preview.json"
    train_df[["id", "cleaned_text", "token_count"]].to_json(
        train_token_path, orient="records", indent=2, force_ascii=False
    )
    val_df[["id", "cleaned_text", "token_count"]].to_json(
        val_token_path, orient="records", indent=2, force_ascii=False
    )
    print(f"Tokenizer used: {DEFAULT_TOKENIZER}")
    print(f"Saved tokenized train preview: {train_token_path}")
    print(f"Saved tokenized validation preview: {val_token_path}")

    print_heading("STEP 5: VALIDATION SUMMARY")
    summary = {
        "raw_rows": int(pd.read_csv(RAW_DATA_PATH).shape[0]),
        "cleaned_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "avg_char_length": round(float(df["char_length"].mean()), 2),
        "avg_word_count": round(float(df["word_count"].mean()), 2),
        "avg_train_token_count": round(float(train_df["token_count"].mean()), 2),
        "avg_validation_token_count": round(float(val_df["token_count"].mean()), 2),
    }
    summary_path = OUTPUT_DIR / "data_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved validation summary: {summary_path}")
    print("\nLO10 pipeline completed successfully.")


if __name__ == "__main__":
    main()
