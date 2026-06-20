#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(
    description="Replace '\\n' (or '/n') in a text file with actual newlines."
)
parser.add_argument("input_file", help="Input text file")
parser.add_argument("output_file", help="Output text file")
parser.add_argument(
    "--pattern",
    choices=["\\n", "/n"],
    default="\\n",
    help="String to replace with newline (default: \\n)",
)

args = parser.parse_args()

with open(args.input_file, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(args.pattern, "\n")

with open(args.output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Converted '{args.input_file}' -> '{args.output_file}'")