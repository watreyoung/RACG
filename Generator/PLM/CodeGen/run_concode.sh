export CUDA_VISIBLE_DEVICES=1
LANG=java
DATADIR=/data/zezhouyang/dataset/concode/concode
OUTPUTDIR=./save/concode
LOGFILE=${OUTPUTDIR}/concode.log
TEEFILE=${OUTPUTDIR}/train.log
# PER_NODE_GPU=YOUR_GPU_NUM       # modify YOUR_GPU_NUM
PER_NODE_GPU=3
EPOCH=10
LOG_STEP=8333
SAVE_STEP=8333

if [ ! -d $OUTPUTDIR ];then
    mkdir -p $OUTPUTDIR
fi

# python  run.py \
python run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_train \
        --do_eval \
        --do_infer \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=6 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=$EPOCH \
        --logging_steps=$LOG_STEP \
        --save_steps=$SAVE_STEP \
        --overwrite_output_dir \
        --seed=42 \
        2>&1 | tee ${TEEFILE}