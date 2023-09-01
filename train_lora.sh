export MODEL_NAME=$1
export VERSION=$2
export SUB_PATH=$3
export DATASET_NAME=$4
export OUTPUT_DATASET_NAME=$5
export WORK_DIR=$6

accelerate launch facechain/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --revision=$VERSION \
    --sub_path=$SUB_PATH \
    --dataset_name=$DATASET_NAME \
    --output_dataset_name=$OUTPUT_DATASET_NAME \
    --caption_column="text" \
    --resolution=512 --random_flip \
    --train_batch_size=1 \
    --num_train_epochs=200 --checkpointing_steps=5000 \
    --learning_rate=1.5e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir=$WORK_DIR \
    --lora_r=4 --lora_alpha=32 \
    --lora_text_encoder_r=32 --lora_text_encoder_alpha=32 --resume_from_checkpoint='fromfacecommon'

    
