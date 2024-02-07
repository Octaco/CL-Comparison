#!/usr/bin/env bash
# Run the Python script with specified arguments

GPU=$1

python3 ../run.py \
    --loss_function INFO_NCE \
    --learning_architecture <your_learning_architecture> \
    --tokenizer_name microsoft/codebert-base \
    --seed 42 \
    --local_rank -2 \
    --log_path ../logging \
    --lang ruby \
    --train_batch_size 7 \
    --eval_batch_size 7 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --n_labels 20 \
    --train_size 0.8 \
    --mrr_path ../data/MRR.txt \
    --log_level INFO \
    --num_of_accumulation_steps 10 \
    --GPU "$GPU"
