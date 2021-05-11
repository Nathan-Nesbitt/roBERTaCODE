# Sets up the environment for development
sh scripts/venv.sh
# Loads all of the data from CodeSearchNet
python scripts/load_data.py
# Cleans the data from CodeSearchNet
python scripts/clean_data.py
