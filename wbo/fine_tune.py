from transformers import (
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    RobertaModel,
    RobertaForSequenceClassification,
    EarlyStoppingCallback,
)
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


"""Script to finetune an RoBERTa PTM on the Wrong Binary Operator data until fully trained."""

model_name = "../roBERTaCODE_python_large/"

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(
    model_name,
    unk_token="<unk>",
    sep_token="</s>",
    cls_token="<s>",
    pad_token="<pad>",
    mask_token="<mask>",
    max_len=510,
)

# Load in the dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "./data/train/full.txt.values",
        "validation": "./data/validation/full.txt.values",
    },
)


def encode(examples):
    return tokenizer(examples["input_ids"], truncation=True, padding="max_length")


train_data = dataset["train"].map(encode, batched=True)
validation_data = dataset["validation"].map(encode, batched=True)
del dataset

print(train_data[0])

# load pretrained model
model = RobertaForSequenceClassification.from_pretrained(model_name)

directory = "roBERTa_WBO"

training_args = TrainingArguments(
    output_dir=directory,
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
)

# Early stoppping callback here. The parameters for it could be tweaked
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    train_dataset=train_data,
    eval_dataset=validation_data,
)

trainer.train()

# Remember to change output directory
trainer.save_model(directory + "python/WBOFullTrain")
