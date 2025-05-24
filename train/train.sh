#!/bin/bash
MODEL="gemma-2-2b-it"

# Define pretrain config
PRETRAIN_ARGS="
    --gradient_accumulation_steps 16 \
    --max_length 512 \
    --batch_size 8 \
    --n_batches 100000000 \
    --n_epochs 1 \
    --save_every 100 \
    --initial_learning_rate 1e-3 \
    --min_learning_rate 1e-8 \
    --warmup_ratio 0.03 \
    --name pretrain \
    --json_file ../data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_dir ../data/LLaVA-Pretrain/images/ \
    --language_model $MODEL \
    --d_type bfloat16"

# Define finetune config
FINETUNE_ARGS="
    --gradient_accumulation_steps 16 \
    --max_length 1024 \
    --batch_size 4 \
    --n_batches 1000000000 \
    --save_every 100 \
    --n_epochs 3 \
    --initial_learning_rate 2e-5 \
    --min_learning_rate 1e-8 \
    --warmup_ratio 0 \
    --name finetune \
    --json_file ../data/LLaVA-Instruct/llava_v1_5_mix665k.json \
    --image_dir ../data/LLaVA-Instruct/ \
    --language_model $MODEL \
    --d_type bfloat16"


echo "Starting pretraining..."
accelerate launch train.py $PRETRAIN_ARGS

# Extract language model name and training name from arguments
LM_NAME=$(echo "$PRETRAIN_ARGS" | grep -o "language_model [^ ]*" | cut -d' ' -f2)
PRETRAIN_NAME=$(echo "$PRETRAIN_ARGS" | grep -o "name [^ ]*" | cut -d' ' -f2)
FINETUNE_NAME=$(echo "$FINETUNE_ARGS" | grep -o "name [^ ]*" | cut -d' ' -f2)

# Copy and rename the last checkpoint for finetuning
cp "weights/projector_${LM_NAME}_${PRETRAIN_NAME}.pth" \
    "weights/projector_${LM_NAME}_${FINETUNE_NAME}.pth"

sleep 10

echo "Starting finetuning..."
accelerate launch train.py $FINETUNE_ARGS

echo "All training completed!"