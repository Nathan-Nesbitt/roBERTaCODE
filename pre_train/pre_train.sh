
# Copyright 2021 Nathan Nesbitt
# See LICENSE.md for more information

# This script runs through all of the different models that can be trained.
# As these can take many days to train to the length nessisary, I have only
# put the first model (java) as the model to train. 

cd ../
# Just incase you forgot to use the venv
. venv/bin/activate

# Runs through the scripts
python3 pre_train/pre-train.py \
    --tokenizer "./tokenizer_java" \
    --language "java" \
    --size "large" \
    --data "data/train_java_large.txt" \
    --validation "data/valid_java_large.txt" \
    --epochs 20 \
    --early_callback \
    --early_stopping_patience 2