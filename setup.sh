# Sets up the environment for development
sh scripts/venv.sh
# Load the venv
. ./venv/bin/activate
# Generates all of the data for a combined, python, and java datasets
python3 scripts/data.py --languages all python java --sizes large