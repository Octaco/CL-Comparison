#!/usr/bin/env bash
# Run the Python script with specified arguments

GPU=$1

python3 ./run.py \
    --loss_function InfoNCE \
    --architecture MoCo \
    --data_path ./data \
    --lang ruby \
    --batch_size 8 \
    --learning_rate 1e-6 \
    --num_train_epochs 5 \
    --train_size 0.8 \
    --data_path ./data/ \
    --log_level DEBUG \
    --num_of_accumulation_steps 16 \
    --num_of_distractors 999 \
    --queue_length 1024 \
    --momentum 0.99 \
    --GPU "$GPU"
