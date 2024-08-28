train='train_data_path.json'
eval='eval_data_path.json'
model='model_name'
n_context=5
output='output'
python generator/fid/train_reader.py \
    --seed 42 \
    --train_data ${train} \
    --eval_data ${eval} \
    --model_name ${model} \
    --per_gpu_batch_size 10 \
    --n_context ${n_context} \
    --name ${output} \
    --checkpoint_dir models/generator/ \
    --eval_freq 217 \
    --accumulation_steps 2 \
    --main_port 30843 \
    --total_steps 2179 \
    --warmup_steps 217
