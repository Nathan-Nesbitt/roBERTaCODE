"""
    Copyright Nathan Nesbitt 2021
"""

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Import the custom trained BERT dataset from
tokenizer = RobertaTokenizerFast.from_pretrained("tokenizer")

# Define the Electra model and config
config = RobertaConfig(vocab_size=32000)
model = RobertaForMaskedLM(config=config)

# Import the dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/train_large.txt",
    block_size=128,
)

# Initialize the data collector
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

# Set all of the training arguments
training_args = TrainingArguments(
    output_dir="./roBERTaCODE_all_large",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_gpu_train_batch_size=24,
    save_steps=10_000,
    save_total_limit=10,
)

trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=dataset
)

trainer.train()

trainer.save_model("./roBERTaCODE_all_large")
