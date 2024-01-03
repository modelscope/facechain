export MODEL_NAME=$1
export VERSION=$2
export SUB_PATH=$3
export IMG_DIR=$4
export WORK_DIR=$5
export USER_ID=$6
export PROMPT_NAME=$7

accelerate launch facechain/train_reward_optimization.py \
    --mixed_precision=no \
    --logdir=./logs/$USER_ID/ \
    --cache_log_file=./log_file.txt \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --face_lora_path=$WORK_DIR \
    --revision=$VERSION \
    --sub_path=$SUB_PATH \
    --sample_batch_size=1 \
    --sample_num_batches_per_epoch=2 \
    --sample_num_steps=40 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=0.0001 \
    --seed=42 \
    --rl_prompt_name=$PROMPT_NAME \
    --use_lora \
    --rank=4 \
    --cfg \
    --allow_tf32 \
    --num_epochs=100 \
    --save_freq=1 \
    --target_image_dir=$IMG_DIR 