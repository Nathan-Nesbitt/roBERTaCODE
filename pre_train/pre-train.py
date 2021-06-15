"""
    Copyright Nathan Nesbitt 2021

    See LICENSE.md for information.

    This file contains the pre-training code. It has a main function that 
    contains the config information, or you can pass the information via CLI.
"""

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
import argparse


class OutputCallback(TrainerCallback):
    """
    This is a custom callback that outputs the current model state.
    This will allow us to visualize the model training as it progresses.
    """

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"Evaluated Epoch {state.epoch}")
        print(f"This epoch resulted in the metric: {self.metrics}")
        print(
            f"The best metric so far is {state.best_metric} on checkpoint {state.best_model_checkpoint }"
        )


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

    validation_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.valid,
        block_size=128,
    )

    # Initialize the data collector
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # Set all of the training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=5,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        train_dataset=dataset,
        eval_dataset=validation_dataset,
    )

    # Added in a callback to output the results for the epochs
    trainer.add_callback(OutputCallback)

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
        "--validation",
        "-v",
        metavar="valid",
        type=str,
        nargs="?",
        help="Location of the validation data file, this is a relative path to the \
            current file. This will default to './data/valid_[size].txt'",
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="output",
        type=str,
        nargs="?",
        help="Location of the output for this run, this is a relative path to the \
            current file. This will default to 'roBERTaCODE_[language]_[size]'",
    )

    args = parser.parse_args()

    if not args.output:
        args.output = "roBERTaCODE_{}_{}".format(args.language, args.size)

    main(args)
