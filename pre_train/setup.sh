# Change back a directory
cd ../

# Activate the environment
. ./venv/bin/activate

# Generates all of the data for a combined, python, and java datasets
python3 pre_train/data.py --languages all python java --sizes large