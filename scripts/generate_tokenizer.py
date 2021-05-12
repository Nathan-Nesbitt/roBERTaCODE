"""
Copyright Nathan Nesbitt 2021-01-16

This creates the tokenizer for each of the data files specified. You can also specify
a set of languages if you want to. 
"""

from tokenizers import ByteLevelBPETokenizer
from glob import glob
import os
import argparse


def main(args):
    generate_tokenizer(args)


def generate_tokenizer(args):

    langs = args.languages

    if "all" in langs:
        langs = ["python", "java", "javascript", "go", "ruby", "php"]

    if args.combined:
        for size in args.sizes:
            lang = "_combined"
            paths = list(glob("data/train{}_{}.txt".format(lang, size)))

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

            os.makedirs("tokenizer{}".format(lang), exist_ok=True)
            tokenizer.save_model("tokenizer{}".format(lang))

    else:
        for language in langs:
            for size in args.sizes:
                lang = "_{}".format(language)

                paths = list(glob("data/train{}_{}.txt".format(lang, size)))

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

                os.makedirs("tokenizer{}".format(lang), exist_ok=True)
                tokenizer.save_model("tokenizer{}".format(lang))


if __name__ == "__main__":
    # Checks to see if there were any params passed via CLI, if not just assign
    # to the defaults in this file.
    parser = argparse.ArgumentParser(
        prog="generate_tokenizer.py",
        description="Generates a tokenizer from a set of data.",
    )

    parser.add_argument(
        "--languages",
        "-l",
        metavar="languages",
        type=str,
        nargs="?",
        help="This is the language(s) you would like to pre-train on.",
        choices=["python", "java", "javascript", "go", "ruby", "php", "all"],
    )

    parser.add_argument(
        "--sizes",
        "-s",
        metavar="sizes",
        type=str,
        nargs="?",
        help="The sizes of the datasets that you are using.",
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
