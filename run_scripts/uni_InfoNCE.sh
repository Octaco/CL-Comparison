#!/usr/bin/env bash
# Run the Python script with specified arguments

GPU=$1

python3 ./run.py \
    --loss_function InfoNCE \
    --architecture Uni \
    --data_path ./data \
    --lang ruby \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --train_size 0.8 \
    --data_path ./data/ \
    --log_level INFO \
    --num_of_accumulation_steps 16 \
    --num_of_distractors 999 \
    --GPU "$GPU"
