"""
Copyright Nathan Nesbitt 2021-01-16

This parses the CodeSearchNet data as JSON and loads all of the program text
into a split one text file, so we can run the tokenizer trainer on the data.

In the original script the goal was to create 3 datasets, as there is 

This is the new version of the file. The original script used in the paper is
marked as clean_data_OLD.py, and works similarly except it does not use the 
circular window to handle the overflow instead opting to trim at exactly 512
tokens. 

The original thesis did not use the sizes, this will be left in but should not
be used for reproducing the original.

This file also reduces the complications around the datasets, as the original 
file required some modifications.

I have left the legacy file as it was used in the original as reference, this
file should be used for all future runs. 
"""

import json
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from glob import glob
import os
import ntpath
import math


def main(args):
    generate_data(args)
    generate_tokenizer(args)


def generate_data(args):
    input_files = {}

    languages = {"go": {}, "java": {}, "javascript": {}, "php": {}, "python": {}}

    # Read in the data based on the language
    for language in args.languages:
        files = []
        location = "data/{}/final/jsonl/train".format(language)
        for path in Path(location).glob("*.jsonl"):
            files.append(location + "/" + path.name)
        input_files[language] = files

    # Calculate the required sizes for small (SQRT(n)), medium (n/2), and large (full)
    for language in args.languages:
        languages[language]["small"] = round(math.sqrt(len(input_files[language])))
        languages[language]["medium"] = round(len(input_files[language]) / 2)
        languages[language]["large"] = round(len(input_files[language]))

    print(languages)

    # This creates the files
    for language, files in input_files.items():
        for size, val in languages[language].items():
            for i in range(val):
                with open(files[i], "r") as input_file:
                    with open(
                        "data/train_{}_{}.txt".format(lang, size), "a"
                    ) as output_file:
                        for line in input_file:
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


def generate_tokenizer(args):

    lang = ""
    if args.languages != "all":
        for i in args.languages:
            lang += "_{}".format(i)

    paths = list(glob("data/train{}_{}.txt".format(lang, args.size)))

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

    os.makedirs("tokenizer_{}".format(lang), exist_ok=True)
    tokenizer.save_model("tokenizer_{}".format(lang))


if __name__ == "__main__":
    # Checks to see if there were any params passed via CLI, if not just assign
    # to the defaults in this file.
    parser = argparse.ArgumentParser(
        prog="clean_data.py",
        description="Transforms the data into an acceptable format for pre-training",
    )

    parser.add_argument(
        "--languages",
        "-l",
        metavar="languages",
        type=str,
        nargs="?",
        help="This is the language you would like to pre-train on.",
        choices=["python", "java", "javascript", "go", "ruby", "php", "all"],
    )

    parser.add_argument(
        "--sizes",
        "-s",
        metavar="sizes",
        type=str,
        nargs="?",
        help="The sizes of tokenizers and datasets that you would like to create.",
        choices=["small", "medium", "large"],
    )

    parser.add_argument(
        "--tokenizer",
        "-t",
        metavar="tokenizer",
        type=str,
        nargs="?",
        help="Location of the tokenizer, this is a relative path to the \
            current file. This defaults to 'tokenizer_[lang]' unless 'all' is \
            specified then it is just 'tokenizer'",
    )

    parser.add_argument(
        "--data",
        "-d",
        metavar="data",
        type=str,
        nargs="?",
        help="Location of the training data file, this is a relative path to the \
            current file. This will default to './data/train_[size].txt'",
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="output",
        type=str,
        nargs="?",
        help="Location of the output for this run, this is a relative path to the \
            current file. This will default to 'roBERTaCODE_[lang]_[size]'",
    )

    args = parser.parse_args()
    main(args)