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
from glob import glob
import math
import argparse
from tqdm import tqdm


def main(args):
    generate_data(args)


def generate_data(args):

    # This is where we store the input files once generated
    input_files = {}

    # This is just a dict of all of the accepted languages and their sizes
    languages = {}

    langs = args.languages

    # If we specify all languages, just initialize all of the langs
    if "all" in langs:
        langs = ["python", "java", "javascript", "go", "ruby", "php"]

    # Read in the files that can be worked with based on the language
    for language in langs:
        languages[language] = {}
        files = []
        location = "data/{}/final/jsonl/train".format(language)
        for path in Path(location).glob("*.jsonl"):
            files.append(location + "/" + path.name)
        input_files[language] = files

    # Calculate the required sizes for small (SQRT(n)), medium (n/2), and large (full)
    for language in langs:
        for size in args.sizes:
            if size == "small":
                languages[language][size] = round(math.sqrt(len(input_files[language])))
            if size == "medium":
                languages[language][size] = round(len(input_files[language]) / 2)
            if size == "large":
                languages[language][size] = round(len(input_files[language]))

    # Read in the files
    for language, files in input_files.items():
        for size, n_files in languages[language].items():
            if args.combined:
                output_file = open("data/train_combined_{}.txt".format(size), "a")
            else:
                output_file = open("data/train_{}_{}.txt".format(language, size), "a")

            with tqdm(total=n_files) as tq:
                tq.set_description("{} Files".format(language))
                for i in range(n_files):
                    with open(files[i], "r") as input_file:
                        for line in input_file:
                            json_data = json.loads(line)
                            # We set the input to be (512 - 3) / 2 tokens to satisfy BERT architechture
                            n = len(json_data["docstring_tokens"])
                            if n > 254:
                                n = 254
                            if args.notext:
                                n = 0
                            output_file.write(
                                " ".join(
                                    ["".join(k) for k in json_data["docstring_tokens"][:n]]
                                )
                            )
                            output_file.write(" ")
                            output_file.write(
                                " ".join(
                                    [
                                        "".join(k)
                                        for k in json_data["code_tokens"][: 509 - n]
                                    ]
                                )
                            )
                            output_file.write("\n")
                        tq.update(1)
            tq.close()


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
        "--combined",
        "-c",
        action="store_true",
        help="Instead of individually creating each languages in it's own file, this script \
            combines all of the languages passed into the model. This is used to create the \
            all languages PTM.",
    )

    parser.add_argument(
        "--notext",
        "-n",
        action="store_true",
        help="This only uses the programming code instead of the programming code and the text. \
            As this was not done in the original paper it is added as an extra parameter, but it \
            can be used to test if code alone is enough to improve performance",
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
    # Gross way of ensuring that inputs are arrays
    a = []
    a.append(args.languages)
    args.languages = a

    a = []
    a.append(args.sizes)
    args.sizes = a
    main(args)
