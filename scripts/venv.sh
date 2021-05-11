# A file that correctly creates a venv in an environment
# installs all of the requirements.txt values

python3 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
