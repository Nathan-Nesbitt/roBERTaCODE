# Just incase you forgot to use the venv
. ../venv/bin/activate

# Sets the dataset directory so we don't run into lock issues
export HF_DATASETS_CACHE=../.cache/datasets/


# Runs through the scripts
python wbo_train.py \
    --model ../roBERTaCODE_java_large \
    --language java \
    --train_data ./data/train/full.txt.values \
    --validation_data ./data/validation/full.txt.values \
    --test_data ./data/test/full.txt.values \
    --output ./models/
    --epochs 40
    --early_callback true
