# Modified version of the original script so it is easier to change the steps
# all you need to do is change the variables after this comment, the rest of 
# the variables will automatically be handled 
lang=java # This is the fine tune language
PTMlang=java # This is the PTM language that you chose, just used for naming 
pretrained_model="../roBERTaCODE_java_large/" # This is where the PTM is stored
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

python scripts/run.py \
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
    --eval_steps $eval_steps