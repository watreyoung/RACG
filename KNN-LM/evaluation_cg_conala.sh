MODEL_PATH=/data/zzyang/models/codet5-base
MODEL=codet5-conala
export CUDA_VISIBLE_DEVICES=1

python -u run_translation_cg.py  \
  --model_name_or_path ${MODEL_PATH} \
  --train_file ./dataset/code_generation/conala/train.json \
  --validation_file ./dataset/code_generation/conala/test.json \
  --test_file ./dataset/code_generation/conala/test.json \
  --per_device_eval_batch_size 64 \
  --output_dir checkpoints-code-generation/${MODEL} \
  --do_eval \
  --predict_with_generate \
  --load original_models/codet5-conala/pytorch_model.bin