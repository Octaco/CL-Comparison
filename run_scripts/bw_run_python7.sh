#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log

source venv/bin/activate

python3 ./run.py \
    --loss_function InfoNCE \
    --architecture Uni \
    --data_path ./data \
    --lang python \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --data_path ./data/ \
    --log_level INFO \
    --num_of_accumulation_steps 8 \
    --num_of_distractors 999