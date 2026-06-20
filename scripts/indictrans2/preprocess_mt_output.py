#!/usr/bin/env python3

import re
import argparse
import unicodedata
from pathlib import Path


def clean_sentence(text):
    """
    Clean MT output text while preserving meaning.
    Intended for BLEU evaluation post-processing.
    """

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Normalize quotes
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

    # Normalize dashes
    text = re.sub(r"[–—−]", "-", text)

    # Remove spaces around dashes
    text = re.sub(r"\s*-\s*", "-", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Ensure exactly one space after punctuation
    text = re.sub(r"([.,!?;:])([^\s])", r"\1 \2", text)

    # Fix apostrophe spacing
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)

    # Remove repeated punctuation
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # Convert abbreviations:
    # W. H. O. -> WHO
    # U. S. -> US
    text = re.sub(
        r"\b(?:[A-Z]\.\s*){2,}",
        lambda m: m.group(0).replace(".", "").replace(" ", ""),
        text
    )

    # Remove duplicated consecutive words
    text = re.sub(
        r"\b(\w+)\s+\1\b",
        r"\1",
        text,
        flags=re.IGNORECASE
    )

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process_file(filepath):
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Reading: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = [clean_sentence(line.rstrip("\\n")) for line in lines]

    output_path = filepath.with_name(
        filepath.stem + "_cleanup" + filepath.suffix
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\\n")

    print(f"Processed {len(cleaned_lines)} lines")
    print(f"Saved cleaned file to:")
    print(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process MT outputs for evaluation."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input MT output file"
    )

    args = parser.parse_args()

    process_file(args.input)


if __name__ == "__main__":
    main()
