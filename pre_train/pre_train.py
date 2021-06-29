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
import logging


class OutputCallback(TrainerCallback):
    """
    This is a custom callback that outputs the current model state.
    This will allow us to visualize the model training as it progresses.
    """

    def __init__(self) -> None:
        super().__init__()

    def on_evaluate(self, args, state, control, **kwargs):
        logging.info(f"Evaluated Epoch {state.epoch}")
        logging.info(f"Current Epoch has metric {self.metrics}")
        logging.info(
            f"The best metric so far is {state.best_metric} on checkpoint {state.best_model_checkpoint }"
        )


class PreTrainer:
    def __init__(
        self,
        tokenizer_location,
        data_location,
        valid_location,
        output_location,
        epochs,
        language,
        size,
        early_callback=False,
        early_stopping_patience=2,
    ) -> None:
        """
        Initializes the Pre-Trainer for RoBERTa. All arguments are fairly straightforward
        but the `early_callback` is used to decide if you want to run it until it reaches
        the epoch count or if you want to kill it first.
        """

        # Define the values for the model
        self.data_location = data_location
        self.valid_location = valid_location
        self.tokenizer_location = tokenizer_location
        self.output_location = output_location
        self.epochs = epochs
        self.language = language
        self.size = size
        self.early_callback = early_callback
        self.early_stopping_patience = early_stopping_patience

        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.tokenizer_location)

        # Create the config file and the model
        self.config = RobertaConfig(vocab_size=32000)
        self.model = RobertaForMaskedLM(config=self.config)

        # Load in the datasets
        self.train_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=self.data_location,
            block_size=128,
        )

        self.validation_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=args.validation,
            block_size=128,
        )

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)

        self.training_args = TrainingArguments(
            output_dir=self.output_location,
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=64,
            save_steps=10_000,
            save_total_limit=5,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
        )

    def train(self):
        trainer = None
        if self.early_callback:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.early_stopping_patience
                    ),
                    OutputCallback(),
                ],
                data_collator=self.data_collator,
                train_dataset=self.train_dataset,
                eval_dataset=self.validation_dataset,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                callbacks=[
                    OutputCallback(),
                ],
                data_collator=self.data_collator,
                train_dataset=self.train_dataset,
                eval_dataset=self.validation_dataset,
            )

        trainer.train(resume_from_checkpoint=True)

        # Save the model
        trainer.save_model(self.output_location)


def main(args):

    trainer = PreTrainer(
        tokenizer_location=args.tokenizer[0],
        data_location=args.data[0],
        valid_location=args.validation[0],
        output_location=args.output[0],
        epochs=args.epochs[0],
        language=args.language[0],
        size=args.size,
        early_callback=args.early_callback,
        early_stopping_patience=args.early_stopping_patience[0],
    )

    trainer.train()


if __name__ == "__main__":
    # Checks to see if there were any params passed via CLI, if not just assign
    # to the defaults in this file.
    parser = argparse.ArgumentParser(
        prog="pre_train.py", description="Pre-Trains the RoBERTa model on code."
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
        nargs=1,
        help="The size of the training set that you are training on. \
            The options are small, medium, large.",
    )

    parser.add_argument(
        "--tokenizer",
        "-t",
        metavar="tokenizer",
        type=str,
        nargs=1,
        help="Location of the tokenizer, this is a relative path to the \
            current file. This defaults to 'tokenizer_[lang]' unless 'all' is \
            specified then it is just 'tokenizer'",
    )

    parser.add_argument(
        "--data",
        "-d",
        metavar="data",
        type=str,
        nargs=1,
        help="Location of the training data file, this is a relative path to the \
            current file. This will default to './data/train_[size].txt'",
    )

    parser.add_argument(
        "--validation",
        "-v",
        metavar="valid",
        type=str,
        nargs=1,
        help="Location of the validation data file, this is a relative path to the \
            current file. This will default to './data/valid_[size].txt'",
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="output",
        type=str,
        nargs=1,
        help="Location of the output for this run, this is a relative path to the \
            current file. This will default to 'roBERTaCODE_[language]_[size]'",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        metavar="epochs",
        type=int,
        nargs=1,
        help="Number of epochs to train the model for.",
    )

    parser.add_argument(
        "--early_callback",
        default=False,
        action="store_true",
        help="This sets the model to stop once it plateaus.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        metavar="early_stopping_patience",
        default=2,
        type=int,
        nargs=1,
        help="How many epochs a model can plateau for before killing.",
    )

    args = parser.parse_args()

    if not args.output:
        args.output = f"roBERTaCODE_{args.language}_{args.size}_{args.epochs}"

    main(args)
