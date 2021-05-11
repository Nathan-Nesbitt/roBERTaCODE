# roBERTaCODE - Pre-Training RoBERTa on Code Summarization
This is the code used to train the model used in Nathan Nesbitt's Honours thesis.
It can be used to pre-train a roBERTa model using code, then fine-tuning it on
the task of code summarization.

The fine-tuning in this repository is based on the CodeBERT repository. 

There have been some changes made, this includes:

1. Removing the circular loading of the data (clean_data_OLD.py)
2. Adding in the ability to break after n epochs in code2nl

## Setup

1. `sh scripts/venv.sh` - Create the virtual environment
2. `python scripts/load_data.py` - Load all of the data from the CodeSearchNet dataset 
3. `python scripts/clean_data.py` - Clean the data from the CodeSearchNet test dataset
    and store it in some line by line files which we will use for pre-training. This also trains
    each one of the tokenizers. 
4. 

1. Run the setup.sh file, this will download the CodeSearchNet dataset, if the venv
    is not defined it will create it and use it, then it will install all of the 
    dependencies defined in requirements.txt.
2. Run the clean_data.py script, this will break the data down into smaller datafiles, 
    and tokenize using Byte Level BPE or Wordpeice depending on the parameters provided. 
3. For each of the models, 3 sizes are required: small, medium, and large. Go into
    the pre_train.py script and change the indicator for each time you run the job
    to ensure that you have a model for each of these sizes. For a dataset of size n
    small is sqrt(n), medium is n/2, and large is n. This simply allows for a comparison
    in the differences in the pre-trained model sizes.
4. For each of the models, the epochs has to be the same, you can decide on this number
    but it must stay consistant for the study to be valid. You can set this inside 
    pre_train.py.
5. Run the pre_train.pbs script as a job, this calls the pre_train.py script which 
    pre-trains the model on the dataset created in step 2 using the parameters defined 
    in steps 3 and 4. This will create a model in the ./models directory with an indicator
    of the size and the epochs.
6. Once you have trained the models, you need to run the setup.sh script in both the code2nl
    and the codesearch directories to set up the environment to have the fine-tuning data
    set up.
7. Now you can queue each of the fine_tune.pbs files inside each of the code2nl and codesearch
    directories. This can be done using the fine_tuner.sh scripts inside of the directories which 
    contains the names of each of the models to fine tune. This will run PBS for you, and fine 
    tunes the model specified inside of the fine_tune.sh script using the names of the files.
8. Once the fine tuning is complete, you need to run the evaluation scripts in each of the 
    code2nl and codesearch directories. These queue up the PBS files in the directories using
    the specified names defined inside of the evaluater.sh which get passed into the evaluate.sh
    script which builds the output for the BLEU model and compares it against the .gold expected
    output file for the code2nl and runs mrr.py on the codesearch model to produce a score. The
    scores will be stores in a scores.txt file.
