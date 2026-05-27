# LO10 Data Handling Evidence

This guide is separate from LO9. It focuses on data collection, cleaning, preprocessing, splitting, tokenization, and validation logs.

## Files

- `src/week10/data/raw_tutor_qa.csv`
- `src/week10/data_handling_pipeline.py`
- `artifacts/week10_lo10/`

## Run Command

```powershell
python src/week10/data_handling_pipeline.py
```

## What The Pipeline Does

1. Loads raw CSV text data
2. Cleans the text
3. Removes duplicates and invalid rows
4. Creates an 80/20 train-validation split
5. Runs tokenizer-based preprocessing
6. Saves cleaned and processed outputs
7. Prints validation statistics

## Screenshots To Take

1. Raw dataset file open in the editor
2. Terminal showing original shape and raw sample rows
3. Terminal showing cleaning results and before/after examples
4. Histogram image `artifacts/week10_lo10/text_length_histogram.png`
5. Terminal showing train/validation split and tokenization output
6. Terminal showing final validation summary JSON
7. File explorer showing the saved outputs in `artifacts/week10_lo10`

## Suggested Report Format

### LO10 Evidence: Data Handling Report

**Objective:**  
To demonstrate text data collection, cleaning, preprocessing, splitting, and validation for a generative AI workflow.

**Raw Data Source:**  
A custom CSV file of tutor-style prompts and text examples was prepared with duplicates, missing rows, URLs, extra punctuation, emojis, mixed casing, and inconsistent spacing.

**Cleaning Steps:**  
- Converted text to lowercase  
- Removed URLs  
- Removed special characters and emojis  
- Normalized whitespace  
- Removed empty rows  
- Removed duplicates  
- Removed very short rows

**Preprocessing Steps:**  
- Calculated text lengths and word counts  
- Created an 80/20 train-validation split  
- Tokenized the cleaned text using a Hugging Face tokenizer  
- Saved cleaned, split, and tokenized outputs

**Validation Evidence:**  
- Original dataset size  
- Cleaned dataset size  
- Number of duplicates removed  
- Number of short rows removed  
- Train and validation sample counts  
- Average character length, word count, and token count

**Result:**  
The raw text data was transformed into a cleaner and more structured dataset suitable for model development and training workflows.
