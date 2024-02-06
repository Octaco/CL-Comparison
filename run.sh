#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonard.remus@uni-ulm.de
#SBATCH --output=./logging/testout.log
#SBATCH --error=./logging/testerr.log

source venv/bin/activate

lang=ruby #fine-tuning a language-specific model for each programming language
idx=0 #test batch idx

python src/run.py