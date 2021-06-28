from transformers import (
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import load_dataset
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging


"""Script to finetune an RoBERTa PTM on the Wrong Binary Operator data until fully trained."""


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


class WBO:
    def __init__(
        self,
        model_name,
        output_directory,
        train_data,
        validation_data,
        test_data,
        epochs=40,
        early_callback=False,
        early_stopping_patience=2,
    ) -> None:

        self.model_name = model_name
        self.output_directory = output_directory
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.epochs = epochs
        self.early_callback = early_callback
        self.early_stopping_patience = early_stopping_patience

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name,
            unk_token="<unk>",
            sep_token="</s>",
            cls_token="<s>",
            pad_token="<pad>",
            mask_token="<mask>",
            max_len=510,
        )

    def train(self):
        # Load in the dataset
        dataset = load_dataset(
            "json",
            data_files={
                "train": self.train_data,
                "validation": self.validation_data,
            },
        )

        train_data = dataset["train"].map(self.encode, batched=True)
        validation_data = dataset["validation"].map(self.encode, batched=True)
        del dataset

        model = RobertaForSequenceClassification.from_pretrained(self.model_name)

        training_args = TrainingArguments(
            output_dir=self.output_directory,
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=32,
            save_steps=10_000,
            save_total_limit=2,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
        )

        trainer = None
        if self.early_callback:
            trainer = Trainer(
                model=model,
                args=training_args,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=2),
                    OutputCallback(),
                ],
                train_dataset=train_data,
                eval_dataset=validation_data,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                callbacks=[
                    OutputCallback(),
                ],
                eval_dataset=validation_data,
            )

        trainer.train()

        trainer.save_model(self.output_directory)

    def encode(self, examples):
        return self.tokenizer(
            examples["input_ids"], truncation=True, padding="max_length"
        )

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        result = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
        with open(f"{self.output_directory}/result.txt", "w") as f:
            f.write(
                f"accuracy:{acc} f1: {f1} precision : {precision} recall : {recall}"
            )
        return result

    def evaluate(
        self,
    ):
        # Load in the dataset
        dataset = load_dataset(
            "json",
            data_files={
                "test": self.test_data,
            },
        )

        test_data = dataset["test"].map(self.encode, batched=True)
        del dataset

        model = RobertaForSequenceClassification.from_pretrained(self.output_directory)

        training_args = TrainingArguments(
            output_dir=self.output_directory,
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=32,
            save_steps=10_000,
            save_total_limit=2,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_data,
            compute_metrics=self.compute_metrics,
        )

        trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fine_tune.py",
        description="Fine-tunes the roBERTaCODE PTM on Wrong Binary Operator",
    )

    parser.add_argument(
        "--model",
        "-m",
        metavar="model",
        type=str,
        nargs=1,
        required=True,
        help="Location of the model, this is a relative path to the \
                current file. We assume that the tokenizer is included in \
                this directory.",
    )

    parser.add_argument(
        "--language",
        "-l",
        metavar="language",
        type=str,
        nargs="?",
        required=True,
        help="This is the language the roBERTa code PTM was trained on. \
            The options are python, java, javascript, go, ruby, php or combined.",
    )

    parser.add_argument(
        "--train_data",
        metavar="train",
        type=str,
        nargs=1,
        help="Location of the training data file, this is a relative path to the \
            current file.",
    )

    parser.add_argument(
        "--validation_data",
        metavar="valid",
        type=str,
        nargs=1,
        help="Location of the validation data file, this is a relative path to the \
            current file.",
    )

    parser.add_argument(
        "--test_data",
        metavar="test",
        type=str,
        nargs=1,
        help="Location of the test data file, this is a relative path to the \
            current file.",
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="output",
        type=str,
        nargs=1,
        help="Location of the output for this run, this is a relative path to the \
            current directory. The language will be appended to this. ",
    )

    parser.add_argument(
        "--epochs",
        metavar="epochs",
        type=int,
        nargs=1,
        help="Number of epochs to train the model for.",
    )

    parser.add_argument(
        "--early_callback",
        metavar="early_callback",
        default=False,
        type=bool,
        nargs=1,
        help="This sets the model to stop once it plateaus.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        "-esp",
        metavar="early_stopping_patience",
        default=2,
        type=int,
        nargs=1,
        help="How many epochs a model can plateau for before killing.",
    )

    parser.add_argument(
        "--train",
        metavar="train",
        default=False,
        type=bool,
        nargs=1,
        help="Should the model be trained?",
    )

    parser.add_argument(
        "--test",
        metavar="test",
        default=False,
        type=bool,
        nargs=1,
        help="Should the model be tested?",
    )

    args = parser.parse_args()

    output = f"{args.output}/{args.language}"

    model = WBO(
        model_name=args.model_name,
        output_directory=output,
        train_data=args.train_data,
        validation_data=args.validation_data,
        test_data=args.test_data,
        epochs=args.epochs,
        early_callback=args.early_callback,
        early_stopping_patience=args.early_stopping_patience,
    )

    if args.train:
        model.train()
    if args.test:
        model.evaluate()
