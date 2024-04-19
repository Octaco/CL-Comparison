#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log

# Run the Python script with specified arguments

GPU=$1

python3 ./run.py \
    --loss_function InfoNCE \
    --architecture MoCo \
    --data_path ./data \
    --lang ruby \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --train_size 0.8 \
    --data_path ./data/ \
    --log_level DEBUG \
    --num_of_accumulation_steps 16 \
    --num_of_distractors 999 \
    --GPU "$GPU"
