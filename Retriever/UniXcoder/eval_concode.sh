# Training
dataset_name=concode

# Evaluating
python run.py \
    --output_dir saved_models/$dataset_name \
    --model_name_or_path /data/zzyang/models/unixcoder-base \
    --do_eval \
    --do_test \
    --train_data_file /data/zzyang/RACG/dataset/$dataset_name/train.json \
    --eval_data_file /data/zzyang/RACG/dataset/$dataset_name/dev.json \
    --test_data_file /data/zzyang/RACG/dataset/$dataset_name/test.json \
    --codebase_file /data/zzyang/RACG/dataset/$dataset_name/train.json \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456