# This is the basic steps based on CodeBERT, using the custom CodeBERT dataset.

# Again incase you forgot to activate the venv
. ../venv/bind/activate

pip3 install gdown
mkdir ../data ../data/code2nl
cd ../data/code2nl
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
cd ../../code2nl