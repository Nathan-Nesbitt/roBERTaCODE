# Just incase you forgot to use the venv
. ../venv/bin/activate

# Sets the dataset directory so we don't run into lock issues
export HF_DATASETS_CACHE=../.cache/datasets/

# Changable model
language="java"
epochs=50
size="large"

# Generated model
model="../roBERTaCODE_${language}_${size}"

# Runs through the scripts
python3 fine_tune.py \
    --model_name $model \
    --language $language \
    --train_data ./data/train/full.txt.values \
    --validation_data ./data/validation/full.txt.values \
    --test_data ./data/test/full.txt.values \
    --output ./models/ \
    --epochs $epochs \
    --train
