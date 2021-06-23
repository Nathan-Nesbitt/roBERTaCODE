"""
    This script fine-tunes a RoBERTa PTM on the task of Wrong Binary Operator 
    (WBO). 
"""

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


def load_data(location, tokenizer):
    """
    Iterates through the input data file adding the source code functions into
    dataValues and the labels into dataLabels

    Makes Sure to seperate [".", "[", "]", "(", ")", "{", "}" ":"]
    Makes sure to keep ["+", "-", "*", "/", "=", "%", "!", "<", ">"]
      together if they are beside eachother (ie. += )
    Otherwise seperates ["+", "-", "*", "/", "=", "%", "!", "<", ">"]

    This is an optimized version of the previous function.
    """
    keep = ["+", "-", "*", "/", "=", "%", "!", "<", ">"]
    separate = [".", "[", "]", "(", ")", "{", "}" ":"]

    reading_file = open(location, "r", encoding="utf-8")
    data_values = []
    data_labels = []

    while True:
        line = reading_file.readline()
        if line == "":
            break
        jsonObject = json.loads(line)
        seperatedFunction = ""
        for element in range(0, len(jsonObject["function"])):
            if jsonObject["function"][element] in separate:
                seperatedFunction += " " + jsonObject["function"][element] + " "
            elif jsonObject["function"][element] in keep:
                if (element > 0) and jsonObject["function"][element - 1] not in keep:
                    seperatedFunction += " "
                seperatedFunction += jsonObject["function"][element]

                if (element + 1) < len(jsonObject["function"]) and jsonObject[
                    "function"
                ][element + 1] not in keep:
                    seperatedFunction += " "
            else:
                seperatedFunction += jsonObject["function"][element]
        data_values.append(
            tokenizer(
                jsonObject["function"],
                padding="max_length",
                max_length=512,
                truncation=True,
            )
        )
        if jsonObject["label"] == "Correct":
            data_labels.append(1)
        else:
            data_labels.append(0)
    reading_file.close()
    return data_values, data_labels


# class that will convert our data into tensors
class codeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {"input_ids": torch.tensor(self.encodings[idx]["input_ids"])}
        item["attention_mask"] = torch.tensor(self.encodings[idx]["attention_mask"])
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Main model name
model_name = "roberta-base"
directory = "roBERTa_WBO"

# Load in the the data for training and validation
train_location = "./data/train/full.txt"
validation_location = "./data/validation/full.txt"

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(
    model_name,
    unk_token="<unk>",
    sep_token="</s>",
    cls_token="<s>",
    pad_token="<pad>",
    mask_token="<mask>",
    max_len=512,
)

# load pretrained model
model = RobertaForSequenceClassification.from_pretrained(model_name)

# The data has to be loaded in first, as we can reallocate the pairs, but not
# the model.
values, labels = load_data(train_location, tokenizer)
dataset = codeDataset(values, labels)

values, labels = load_data(validation_location, tokenizer)
validation = codeDataset(values, labels)

del values, labels

training_args = TrainingArguments(
    output_dir=directory,
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=64,
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
    train_dataset=dataset,
    eval_dataset=validation,
)

trainer.train()

# Remember to change output directory
trainer.save_model(directory + "python/WBOFullTrain")
