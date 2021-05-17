# Sets up the environment for development
sh scripts/venv.sh
# Load the venv
. ./venv/bin/activate
# Loads all of the data from CodeSearchNet
python3 scripts/load_data.py
# Cleans the data from CodeSearchNet
python3 scripts/clean_data.py --languages all --sizes large -c
python3 scripts/clean_data.py --languages java --sizes large
python3 scripts/clean_data.py --languages python --sizes large
# Creates the tokenizers for each of the datasets
python3 scripts/generate_tokenizer.py --languages all --sizes large -c
python3 scripts/generate_tokenizer.py --languages python --sizes large
python3 scripts/generate_tokenizer.py --languages java --sizes large