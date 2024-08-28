lang=java #programming language
lr=5e-5
batch_size=256
accm_steps=1
beam_size=3
source_length=512
target_length=150
data_dir=../dataset/concode
output_dir=saved_models/concode/$lang
test_file=$data_dir/test.json
pretrained_model=microsoft/unixcoder-base

python run.py \
--do_test \
--lang $lang \
--model_name_or_path $pretrained_model \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--gradient_accumulation_steps $accm_steps \
--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log

