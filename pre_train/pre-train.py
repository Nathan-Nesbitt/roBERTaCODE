"""
    Copyright Nathan Nesbitt 2021

    See LICENSE.md for information.

    This file contains the pre-training code. It has a main function that 
    contains the config information, or you can pass the information via CLI.
"""

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse


def main(args):

    # Import the custom trained tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)

    # Define the model
    config = RobertaConfig(vocab_size=32000)
    model = RobertaForMaskedLM(config=config)

    # Import the dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.data,
        block_size=128,
    )

    # Initialize the data collector
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # Set all of the training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_gpu_train_batch_size=24,
        save_steps=10_000,
        save_total_limit=10,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    # Save the mode
    trainer.save_model("./roBERTaCODE_{}_{}".format(args.language, args.size))


if __name__ == "__main__":
    # Checks to see if there were any params passed via CLI, if not just assign
    # to the defaults in this file.
    parser = argparse.ArgumentParser(
        prog="pre-train.py", description="Pre-Trains the RoBERTa model on code."
    )

    parser.add_argument(
        "--language",
        "-l",
        metavar="language",
        type=str,
        nargs="?",
        help="This is the language you would like to pre-train on. \
            The options are python, java, javascript, go, ruby, php or all.",
    )

    parser.add_argument(
        "--size",
        "-s",
        metavar="size",
        type=str,
        nargs="?",
        help="The size of the training set that you are training on. \
            The options are small, medium, large.",
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

    """
        Overrides arguments from CLI if not existant. Default values are:
        
            language: java 
            size: large
            tokenizer: tokenizer_java
            data: /data/train_java.txt
            output: roBERTaCODE_java_large
        
        You can either pass these to the script via the CLI or you can modify
        them below. Realistically you should only have to modify the 
    """
    if not args.language:
        args.language = "java"
    if not args.size:
        args.size = "large"
    if not args.tokenizer:
        if args.language == "all":
            args.tokenizer = "tokenizer"
        else:
            args.tokenizer = "tokenizer_{}".format(args.language)
    if not args.data:
        args.data = "./data/train_{}.txt".format(args.size)
    if not args.output:
        args.output = "roBERTaCODE_{}_{}".format(args.language, args.size)

    main(args)