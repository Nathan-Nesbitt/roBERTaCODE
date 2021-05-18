
# First thing we load in the venv to ensure you didn't forget 

. ../venv/bin/activate

# Modified version of the original script so it is easier to change the steps
# all you need to do is change the variables after this comment, the rest of 
# the variables will automatically be handled 

lang=java # This is the fine tune language
PTMlang=java # This is the PTM language that you chose, just used for naming 
pretrained_model="../roBERTaCODE_java_large/" # This is where the PTM is stored, or roberta-base if you don't have a PTM
epochs=40 # This is the number of epochs, view the readme to understand why this is 40 

# These are variables that can be changed if you want to change the experiment
# but they will not allow for repeat.

lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others (CODEBERT comment)
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others (CODEBERT comment)
data_dir=../data/code2nl/CodeSearchNet

# These are variables that are created from previous variables

output_dir="model/PTM_${PTMlang}_epochs_${epochs}_finetune_${lang}"
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl 

# Main script

if python3 scripts/run.py \
    --do_train \
    --do_eval \
    --model_type roberta \
    --model_name_or_path $pretrained_model \
    --train_filename $train_file \
    --dev_filename $dev_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --learning_rate $lr \
    --train_steps $train_steps \
    --num_train_epochs $epochs \
    --eval_steps $eval_steps ; then

    # Running custom BLEU 

    # Change into the proper directory
    cd $output_dir

    # Incase you haven't loaded in git on the system
    # module load git

    # Had some issues with SSL certs on the system, this downloads punkt from source.
    # curl https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip --output punkt.zip
    # unzip punkt.zip


    # Sets the original output files. These are the test files instead of the validation.
    targets="test_1.gold"
    predictions="test_1.output"

    # Removes the numbers from the files
    sed -e 's/^[0-9][0-9]*[[:space:]][[:space:]]*//' $targets > targets.out
    sed -e 's/^[0-9][0-9]*[[:space:]][[:space:]]*//' $predictions > predictions.out

    # Installs the BLEU scoring scripts
    pip3 install git+https://github.com/Maluuba/nlg-eval.git@master

    # Evaluates
    nlg-eval --setup
    nlg-eval --hypothesis=targets.out --references=predictions.out > scores.txt

else 
    echo "Error occured before model trained. No output files generated."
fi