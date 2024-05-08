#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log


source venv/bin/activate

python ./run.py --sweep_id p1hoblcn --num_of_runs 2