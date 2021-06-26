# Just incase you forgot to use the venv
. ../venv/bin/activate

# Sets the dataset directory so we don't run into lock issues
export HF_DATASETS_CACHE=../.cache/datasets/


# Runs through the scripts
python3 wbo_train.py