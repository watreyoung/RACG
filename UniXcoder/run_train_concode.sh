lang=java #programming language
lr=5e-5
batch_size=32
accm_steps=1
beam_size=3
source_length=512
target_length=150
data_dir=../dataset/concode
output_dir=saved_models/concode/$lang
train_file=$data_dir/train.json
dev_file=$data_dir/dev.json
epochs=3
pretrained_model=microsoft/unixcoder-base

mkdir -p $output_dir
python run.py \
--do_train \
--do_eval \
--lang $lang \
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
--gradient_accumulation_steps $accm_steps \
--num_train_epochs $epochs 2>&1| tee $output_dir/train.log