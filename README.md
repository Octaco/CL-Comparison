# General
This repository contains the code for my Bachelors Thesis "Comparative Analysis of Contrastive Loss Functions for
Semantic Code Search in Various Architectures".

To understand which formulations and frameworks work best for the task of natural language code search,
we systematically evaluate different contrastive loss setups. The
loss functions Contrastive Loss, Triplet Loss, and InfoNCE are
combined with a Uni-encoder, a bi-encoder, and a momentum
contrastive learning architecture. Each combination is evaluated
on six different programming languages using the well-known
CodeSearchNet dataset with a pre-trained CodeBERT as the base
encoder model. Additionally, we evaluate the generalization ca-
pabilities of these setups using the StatCodeSearch dataset

# Requirements
- **Python 3.10 or higher**. The software is written in Python 3.10, and it is not guaranteed to work with older versions of Python.
- **Nvidia GPU** 

# Installation

1. Clone the repository from https://github.com/Octaco/CL-Comparison.git
2. go to the project directory and create a virtual environment 
```bash
python3 -m venv venv
```
3. Activate the virtual environment
```bash
source venv/bin/activate
```
4. Install the required packages
```bash
pip install -r requirements.txt
```
### Optional: visualizing the embeddings
5. create a directory plots in the data folder:
```bash
mkdir data/plots
```
6. create a directory for the dataset you want to test in the plots folder(for example python):
```bash
mkdir data/plots/python
```
# Usage
### Basic Usage
1. Run the following command to train a Uni-encoder model with InfoNCE loss on the python dataset:
```bash
sh run_scripts/basic_run.sh
```

### Configure the script to run a different model, dataset, loss or hyperparameters:
You can configure the script to run a different model, dataset, loss or hyperparameters 
by changing the arguments in the run_scripts/basic_run.sh file, or writing your own script.

Possible arguments:
- `--loss_function`: The loss function to use. Possible values: `InfoNCE`, `triplet`, `ContrastiveLoss`. Default: `InfoNCE`.
- `--architecture`: The model to use. Possible values: `Uni`, `Bi`, `MoCo`. Default: `Uni`.
- `--tokenizer_name`: The tokenizer to use. Default: `microsoft/codebert-base`.
- `--seed`: The random seed to use. Default: `42`.
- `--log_path`: The path to save the logs. Default: `./logging`.
- `--lang`: The dataset to use. Possible values: `python`, `java`, `javascript`, `go`, `php`, `ruby`. Default: `python`.
- `--batch_size`: The batch size to use. Default: `16`.
- `--learning_rate`: The learning rate to use. Default: `1e-5`.
- `--num_train_epochs`: The number of epochs to train. Default: `5`.
- `--momentum`: The momentum to use. Default: `0.999`.
- `--train_size`: The proportion of the training set, which is used for training. Default: `0.8`.
- `--data_path`: The path to the dataset. Default: `./data/`.
- `--log_level`: The log level to use. Possible values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default: `INFO`.
- `--num_of_accumulation_steps`: The number of accumulation steps to use. Default: `16`.
- `--num_of_negative_samples`: The number of negative samples to use. Default: `15`.
- `--num_of_distractors`: The number of distractors to use. Default: `99`.
- `--queue_length`: The length of the queue to use for the momentum encoder. Default: `4096`.
- `--GPU`: use this argument to specify the GPU to use. 
- `--do_generalisation`: use this argument to enable generalisation testing. Default: `True`.
- `--do_validation`: use this argument to enable validation. Default: `True`.
- `--do_visualization`: use this argument to enable visualisation. Default: `True`.
