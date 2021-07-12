# Copyright 2021 Nathan Nesbitt
# See LICENSE.md for more information

# This script runs through all of the different models that can be trained.
# As these can take many days to train to the length nessisary, I have only
# put the first model (java) as the model to train. 

cd ../
# Just incase you forgot to use the venv
. venv/bin/activate

# Change these values based on the model and size you want
size="large"
language="combined"
epochs=50

# These are generated based on the previous variables
tokenizer="./tokenizer_${language}_${size}"
data="data/train_${language}_${size}.txt"
validation="data/valid_${language}_${size}.txt"

# Runs through the scripts, if you don't want it to halt 
# after plateau remove early_callback
python pre_train/pre_train.py \
    --tokenizer $tokenizer \
    --language $language \
    --size $size \
    --data $data \
    --validation $validation \
    --epochs $epochs \
    --early_callback