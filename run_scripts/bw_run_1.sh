#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log


source venv/bin/activate

python ./run.py --sweep_id bp4d2igb --num_of_runs 2