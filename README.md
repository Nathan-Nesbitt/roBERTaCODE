# RoBERTaCODE - Pre-Training RoBERTa on Code Summarization
This is the code used to train the model used in Nathan Nesbitt's Honours thesis
on code summarization using Transformer PTMs. It can be used to pre-train a 
roBERTa model using code, then fine-tuning it on the task of code summarization.

The fine-tuning in this repository is based on the CodeBERT repository. 

The original thesis code has been re-written to make it easier to understand and
run.

## Requirements
- Python3
- A GPU
- 20GB or so of storage. (Depending on what dataset you download)

## Steps

The default scripts are configured to pre-train similar to the thesis. There
are 3 main models:

- Python PTM
- Java PTM
- All PTM

These 3 models are then fine-tuned on the downstream task of code2nl or Code
Summarization as defined in [CodeBERT](https://github.com/microsoft/CodeBERT).

All of the `.sh` scripts are defined to run the models as they were run for 
the thesis, and can be modified to produce different languages.

### Pre-Training

1. `sh setup.sh` - Sets up for pre-training using the [setup script](setup.sh). This runs a set of scripts that can be found under the [scripts directory](./scripts/). This sets up a virtual environment for python, installs all of the requirements, downloads the data, spreads it into the proper structure, then creates the appropriate tokenizers.

2. `sh pre-train.sh` - Run the [pre-training script](./pre_train/pre-train.sh). This will run 
    on the default values for one of the 3 models. These take a really long time (3 days at a
    minimum running on a GPU) so only 1 is generated in the script.

### Fine-Tuning

#### Code2NL
Look at the [code2nl README](./code2nl/README.md) for full steps.