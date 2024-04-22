#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log

sweep_id=$1

source venv/bin/activate

python ./run.py --sweep_id $sweep_id