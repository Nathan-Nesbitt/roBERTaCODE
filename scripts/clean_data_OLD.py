"""
Name:   clean_data.py
Author: Nathan Nesbitt
Date:   2021-01-16

Outline:

This parses the CodeSearchNet data as JSON and loads all of the program text
into a split one text file, so we can run the tokenizer trainer on the data.

"""

import json
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from glob import glob
import os
import ntpath
import math

input_files = {}

# languages = {"go": {}, "java": {}, "javascript": {}, "php": {}, "python": {}}

lang = "_php"

languages = {"java": {}}

# Read in the data based on the language
for language in languages:
    files = []
    location = "data/" + language + "/final/jsonl/train"
    for path in Path(location).glob("*.jsonl"):
        files.append(location + "/" + path.name)
    input_files[language] = files

# Calculate the required sizes for small (SQRT(n)), medium (n/2), and large (full)
for language, files in input_files.items():
    languages[language]["small"] = round(math.sqrt(len(files)))
    languages[language]["medium"] = round(len(files) / 2)
    languages[language]["large"] = round(len(files))

print(languages)

"""
Creates 3 files: train_small, train_medium, train_large with the sizes defined 
before. We also define a window, as the max input size is often shorter than 
the length of each code segment. This makes it not possible to handle the full
input so it is split each 512 inputs into a new line, with a negative offset
to re-introduce the pairs.
"""

offset = 4
for language, files in input_files.items():
    for size, val in languages[language].items():
        for i in range(val):
            with open(files[i], "r") as input_file:
                with open(
                    "data/train" + lang + "_" + size + ".txt", "a"
                ) as output_file:
                    for line in input_file:
                        n_token = 0
                        docstring = 0
                        code = 0
                        json_data = json.loads(line)
                        for docstring in range(len(json_data["docstring_tokens"])):
                            if n_token % 512 == 0 and n_token != 0:
                                output_file.write("\n")
                                docstring -= offset
                            output_file.write(
                                json_data["docstring_tokens"][docstring] + " "
                            )
                            n_token += 1
                        for code in range(len(json_data["code_tokens"])):
                            if n_token % 512 == 0 and n_token != 0:
                                output_file.write("\n")
                                code = code - offset
                            output_file.write(json_data["code_tokens"][code] + " ")
                            n_token += 1
                        output_file.write("\n")


# Create the tokenizer
paths = list(glob("data/train" + lang + "_large.txt"))

tokenizer = ByteLevelBPETokenizer(lowercase=False)

tokenizer.train(
    files=paths,
    vocab_size=32000,
    min_frequency=3,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

os.makedirs("tokenizer" + lang, exist_ok=True)
tokenizer.save_model("tokenizer" + lang)
