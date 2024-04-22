#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log

#source venv/bin/activate

GPU=$1

python ./run.py --sweep_id onename-org/Bachelor_Thesis/27i51dfx --GPU $GPU