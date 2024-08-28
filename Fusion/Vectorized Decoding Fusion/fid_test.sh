test_data='test_data_path.json'
n_context=5
output='output'
python generator/fid/test_reader_simple.py \
    --model_path models/generator/${output}/checkpoint/best_dev \
    --tokenizer_name models/generator/codet5-base \
    --eval_data  ${eval}  \
    --per_gpu_batch_size 10 \
    --n_context ${n_context} \
    --name ${output}_test \
    --checkpoint_dir models/generator  \
    --result_tag test_same \
    --main_port 81692