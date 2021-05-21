# Fine-Tuning
This folder contains all of the necessary scripts to fine-tune a model on the code2nl task 
also known as Code Summarization.

## Running

1. Step 1 is to set up the model you would like to fine tune. In this research I used RoBERTa
2. You can pre-train the model using the main script, or you can just use the plain RoBERTa 
    model and pass it in. 
3. Open the [fine-tune.sh](fine-tune.sh) script. This is a template script that 
    outlines the required parameters for the `run.py` script provided in the CodeBERT repo. 
    To fine-tune a model you simply need to change the following parameters

    - lang - This is the downstream language. It can 
        be python, java, go, javascript, ruby, php

    - PTMlang - This is just used for naming, set it to 
        be the language you used for pre-training.

    - pretrained_model - Path to your pre-trained model

    - epochs - Number of epochs to train on. This is default set as a large number which
        automatically saves the best possible model. The reason for this is outlined in 
        my thesis as the CodeBERT repository does not use partial training.  

4. Run the `setup.sh` script to download the datasets.
5. Move the tokenizer files for the language into the `pretrained_model` directory.
    This means that you should have the `merges.txt` and `vocab.json` in the 
    `roBERTaCODE_{language}_{size}` directory.
5. Run the main script for training `fine-tune.sh`

## Differences from CodeBERT repository

There are a couple of differences, as the original CodeBERT paper focused on only fully 
training the models I had to add in the functionality to break after `n` epochs.

The original scripts output 2 files for the testing, you should be aware that one of these
files is `validation` data and one is `testing` data. The extra scripts for calculating the
BLEU scores take this into account and only calculate the `test` scores, *BUT* if you are
looking at the raw output (`test_{}.gold` and `test_{}.output`) the first file is from the 
`validation` dataset and the second file is from the `test` dataset. 

We also provide the full BLEU metrics (1-4) along with METEOR, ROUGE_L, and CIDEr.
The original CodeBERT repo used the smoothed BLEU score.