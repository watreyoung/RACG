MODEL_PATH=/data/zzyang/models/codet5-base
MODEL=codet5-conala
export CUDA_VISIBLE_DEVICES=0

python -u run_translation_cg.py  \
  --model_name_or_path ${MODEL_PATH} \
  --train_file ./dataset/code_generation/conala/train.json \
  --validation_file ./dataset/code_generation/conala/dev.json \
  --test_file ./dataset/code_generation/conala/test.json \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --output_dir checkpoints-code-generation/${MODEL} \
  --dstore_dir checkpoints-code-generation/${MODEL} \
  --save_knnlm_dstore \
  --do_eval \
  --eval_subset train \
  --load original_models/codet5-conala/pytorch_model.bin


# MODEL_PATH=/data/zzyang/models/codet5-base
# MODEL=codet5
# export CUDA_VISIBLE_DEVICES=1

# python -u run_translation_cg.py  \
#   --model_name_or_path ${MODEL_PATH} \
#   --train_file ./dataset/code_generation/conala/train.json \
#   --validation_file ./dataset/code_generation/conala/dev.json \
#   --test_file ./dataset/code_generation/conala/test.json \
#   --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
#   --output_dir checkpoints-code-generation/${MODEL} \
#   --dstore_dir checkpoints-code-generation/${MODEL} \
#   --save_knnlm_dstore \
#   --do_eval \
#   --eval_subset train