import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader,TensorDataset
from datasets import load_dataset

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, einsum

from torch import cuda

import logging
from tqdm import tqdm
import os.path as osp
import os

import matplotlib.pyplot as plt

from sklearn import metrics

from timeit import default_timer as timer



from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

###############################################################################
# Experiment Setup
###############################################################################
dataset = 'codebert-base'
language = 'ruby'
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
code_search_dataset = load_dataset('code_search_net', language)

train_data = code_search_dataset['train']
test_data = code_search_dataset['test']

# columns in the dataset: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string',
# 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens',
# 'split_name', 'func_code_url']"

function_code = train_data['func_code_string']
function_documentation = train_data['func_documentation_string']

function_code_test = test_data['func_code_string']
function_documentation_test = test_data['func_documentation_string']





################################################################################
# Create DataFrames
################################################################################

train_labels = []
test_labels = []




train_df1 = pd.DataFrame()
train_df1['text'] = function_documentation

test_df1 = pd.DataFrame()
test_df1['text'] = function_documentation

train_df2 = pd.DataFrame()
train_df2['text'] = function_code

test_df2 = pd.DataFrame()
test_df2['text'] = function_code


################################################################################
# Create Custom Dataset
################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")



class CustomDataset(TensorDataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        # self.targets = dataframe.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())


        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        #token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }



train_size = 0.8
train_dataset = train_df1.sample(frac=train_size, random_state=200)
valid_dataset = train_df1.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = test_df1.reset_index(drop=True)

train_dataset2 = train_df2.sample(frac=train_size, random_state=200)
valid_dataset2= train_df2.drop(train_dataset.index).reset_index(drop=True)
train_dataset2 = train_dataset.reset_index(drop=True)
test_dataset2 = test_df2.reset_index(drop=True)


print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VAL Dataset: {}".format(valid_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
#print("len:" , testing_set.__len__())
#print(testing_set)

training_set2 = CustomDataset(train_dataset2, tokenizer, MAX_LEN)
validation_set2 = CustomDataset(valid_dataset2, tokenizer, MAX_LEN)
testing_set2 = CustomDataset(test_dataset2, tokenizer, MAX_LEN)
#print("len2:" , testing_set2.__len__())

#print(testing_set.__getitem__(0))
###############################################################################
# load model and tokenizer
###############################################################################

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }

print()
#dataset_train = torch.utils.data.TensorDataset(training_set, training_set2)
#dataset_test = torch.utils.data.TensorDataset(testing_set, testing_set2)
#dataset_validation = torch.utils.data.TensorDataset(validation_set, validation_set2)

dataloader_train = DataLoader(training_set, **train_params)




device = 'cuda' if cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.ERROR)


###############################################################################
# model
###############################################################################

model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)

for index, item in enumerate(dataloader_train):
    id = item["ids"]
    mask = item["mask"]
    output = model(id, mask)
    print(output[0].shape)
    break



###############################################################################
# NL-PL embeddings
###############################################################################

# n1_tokens=tokenizer.tokenize("return maximum value")
# code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
# tokens = [tokenizer.cls_token] + n1_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
# tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
# context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
# print(tokens)