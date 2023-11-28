import torch
import pandas as pd
import numpy as np




from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

###############################################################################
# Experiment Setup
###############################################################################
dataset = 'codebert-base'   
n_labels = 20 #[20,8,52,23,2]
num_epochs = 10

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
LEARNING_RATE = 1e-5

###############################################################################
# Load Data
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)


################################################################################
# Create DataFrames
################################################################################

train_labels = []
test_labels = []
train_data = []
test_data = []



train_df = pd.DataFrame()
train_df['text'] = train_data
train_df['labels'] = train_labels

test_df = pd.DataFrame()
test_df['text'] = test_data
test_df['labels'] = test_labels

print("Number of train texts ",len(train_df['text']))
print("Number of train labels ",len(train_df['labels']))
print("Number of test texts ",len(test_df['text']))
print("Number of test labels ",len(test_df['labels']))

###############################################################################
# create train and test dataset
###############################################################################





###############################################################################
# NL-PL embeddings
###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
n1_tokens=tokenizer.tokenize("return maximum value")
code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
tokens = [tokenizer.cls_token] + n1_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
print(tokens)