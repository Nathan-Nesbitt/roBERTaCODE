# Only defines these variables to be a default if you call the script directly
if [[ -z $pre_trained_model ]]; then
    pretrained_model="../roBERTaCODE_all_large/"
fi
if [[ -z $lang ]]; then
    lang=java
fi
epochs=40
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../data/code2nl/CodeSearchNet
output_dir="model/allPTM_epochs${epochs}_finetune_${lang}"
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others

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
