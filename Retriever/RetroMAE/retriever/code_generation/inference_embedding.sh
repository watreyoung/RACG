# 运行这个之前需要手动创建mapping.txt
# python -m torch.distributed.launch --nproc_per_node 1 \
# -m 
python /data/zezhouyang/Retrieve/RetroMAE/src/bi_encoder/run.py \
--output_dir /data/zezhouyang/Retrieve/RetroMAE/examples/retriever/msmarco/HS/retromae_msmarco_passage_fintune \
--model_name_or_path /data/zezhouyang/models/RetroMAE_MSMARCO_finetune  \
--corpus_file /data/zezhouyang/Retrieve/RetroMAE/examples/retriever/msmarco/HS/heartstone_train_retrieve/train \
--passage_max_len 128 \
--fp16  \
--do_predict \
--prediction_save_path /data/zezhouyang/Retrieve/RetroMAE/examples/retriever/msmarco/HS/results_train/ \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100 

# python -m torch.distributed.launch --nproc_per_node 1 \
# -m bi_encoder.run \
python /data/zezhouyang/Retrieve/RetroMAE/src/bi_encoder/run.py \
--output_dir /data/zezhouyang/Retrieve/RetroMAE/examples/retriever/msmarco/HS/retromae_msmarco_passage_fintune \
--model_name_or_path /data/zezhouyang/models/RetroMAE_MSMARCO_finetune  \
--corpus_file /data/zezhouyang/Retrieve/RetroMAE/examples/retriever/msmarco/HS/heartstone_train_retrieve/test \
--passage_max_len 128 \
--fp16  \
--do_predict \
--prediction_save_path /data/zezhouyang/Retrieve/RetroMAE/examples/retriever/msmarco/HS/results_test/ \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100

# python -m torch.distributed.launch --nproc_per_node 1 \
# -m bi_encoder.run \
# --output_dir retromae_msmarco_passage_fintune \
# --model_name_or_path Shitao/RetroMAE_MSMARCO_finetune  \
# --corpus_file ./heartstone_train_retrieve/dev \
# --passage_max_len 128 \
# --fp16  \
# --do_predict \
# --prediction_save_path results_dev/ \
# --per_device_eval_batch_size 256 \
# --dataloader_num_workers 6 \
# --eval_accumulation_steps 100 