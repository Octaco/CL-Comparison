#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log

source venv/bin/activate

python3 ./run.py \
    --loss_function ContrastiveLoss \
    --architecture MoCo \
    --data_path ./data \
    --lang javascript \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --data_path ./data/ \
    --log_level INFO \
    --num_of_accumulation_steps 128 \
    --num_of_distractors 999 \
    --queue_length 1024 \
    --momentum 0.99