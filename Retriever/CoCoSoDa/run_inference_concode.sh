lang=java
current_time=$(date "+%Y%m%d%H%M%S")

code_length=256
nl_length=256

model_type=base #"base", "cocosoda" 
moco_k=1024
moco_m=0.999
lr=2e-5
moco_t=0.07

batch_size=64
max_steps=1000
save_steps=100
aug_type_way=random_replace_type 
data_aug_type=random_mask

base_model=DeepSoftwareAnalytics/CoCoSoDa
epoch=5
# echo ${base_model}
CUDA_VISIBLE_DEVICES=2
# exit 111


function inference () {
output_dir=./saved_models/fine_tune/concode
mkdir -p $output_dir
echo ${output_dir}
 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python my_inference.py   --eval_frequency  100 \
    --moco_m ${moco_m} --moco_t ${moco_t}  \
    --model_type ${model_type} \
    --output_dir ${output_dir}  \
    --data_aug_type ${data_aug_type} \
    --moco_k ${moco_k} \
    --config_name=${base_model}  \
    --model_name_or_path=${base_model} \
    --tokenizer_name=${base_model} \
    --lang=$lang \
    --do_test \
    --train_data_file=dataset/concode/train.json \
    --eval_data_file=dataset/concode/dev.json  \
    --test_data_file=dataset/concode/test.json \
    --codebase_file=dataset/concode/codebase.json \
    --num_train_epochs ${epoch} \
    --code_length ${code_length} \
    --nl_length ${nl_length} \
    --train_batch_size ${batch_size} \
    --eval_batch_size 64 \
    --learning_rate ${lr} \
    --seed 123456 2>&1| tee ${output_dir}/inference.log
}

inference
