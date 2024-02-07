import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from info_nce import InfoNCE
from transformers import RobertaModel, RobertaTokenizer
from datasets import load_dataset

from datetime import datetime
import time
from tqdm import tqdm

LOSS_FUNCTIONS = ['triplet', 'INFO_NCE', 'Soft_nearest_neighbour']
LEARNING_ARCHITECTURES = ['SimCLR', 'SimSiam', 'MoCo']


class CustomDataset(TensorDataset):

    def __init__(self, dataframe, args):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.doc = dataframe.doc
        self.code = dataframe.code
        # self.targets = dataframe.labels
        self.max_len = args.MAX_LEN

    def __len__(self):
        assert len(self.doc) == len(self.code)
        return len(self.doc)

    def __getitem__(self, index):
        doc = str(self.doc[index])
        doc = " ".join(doc.split())

        code = str(self.code[index])
        doc_inputs = self.tokenizer.encode_plus(
            doc,
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        )
        doc_ids = doc_inputs['input_ids']
        doc_mask = doc_inputs['attention_mask']

        code_inputs = self.tokenizer.encode_plus(
            code,
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        )

        code_ids = code_inputs['input_ids']
        code_mask = code_inputs['attention_mask']

        return {
            'doc_ids': torch.tensor(doc_ids, dtype=torch.long),
            'doc_mask': torch.tensor(doc_mask, dtype=torch.long),
            'code_ids': torch.tensor(code_ids, dtype=torch.long),
            'code_mask': torch.tensor(code_mask, dtype=torch.long),
        }


def load_data(args):
    code_search_dataset = load_dataset('code_search_net', args.lang)

    # train_data
    train_data = code_search_dataset['train']

    function_code = train_data['func_code_string']
    function_documentation = train_data['func_documentation_string']

    train_df = pd.DataFrame()
    train_df['doc'] = function_documentation
    train_df['code'] = function_code

    # test_data
    test_data = code_search_dataset['test']

    function_code_test = test_data['func_code_string']
    function_documentation_test = test_data['func_documentation_string']

    test_df = pd.DataFrame()
    test_df['doc'] = function_documentation_test
    test_df['code'] = function_code_test

    return train_df, test_df


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def calculate_mrr_from_distances(distances_lists):
    ranks = []
    for batch_idx, predictions in enumerate(distances_lists):
        correct_score = predictions[0]
        scores = np.array([prediction for prediction in predictions])
        rank = np.sum(scores <= correct_score)
        ranks.append(rank)
    mean_mrr = np.mean(1.0 / np.array(ranks))

    return mean_mrr


def write_mrr_to_file(args, mrr, test=False):
    mrr_old = ""
    with open(args.mrr_path, "r") as file:
        mrr_old = file.read()

    with open(args.mrr_path, "w") as file:
        now = datetime.now().strftime("%d/%m/%Y%H:%M")

        if test:
            mrr_new = f"{mrr_old}(test) {now}:{args.lang},{args.loss_function} {mrr}\n"
        else:
            mrr_new = f"{mrr_old}{now}:{args.lang},{args.loss_function} {mrr}\n"
        file.write(mrr_new)


def train(args, loss_formulation, model, optimizer, training_set):
    logging.info("Start training ...")
    print("Start training ...")
    for epoch in tqdm(range(1, args.num_train_epochs + 1)):
        logging.info("training epoch %d", epoch)

        all_losses = []

        batch_size = args.train_batch_size
        train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

        for idx, batch in enumerate(train_dataloader):

            if batch['code_ids'].size(0) < args.train_batch_size:
                logging.debug("continue")
                continue

            # query = doc
            query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': query_id, 'attention_mask': query_mask}
            query = model(**inputs)[1]  # using pooled values

            logging.debug(f"idx: {idx}")
            logging.debug(f"code_ids: {batch['code_ids'].shape}")
            logging.debug(f"code_ids_0: {batch['code_ids'][0].shape}")

            logging.debug(f"mask: {batch['code_mask'].shape}")
            logging.debug(f"mask_0: {batch['code_mask'][0].shape}")

            code_list = [(batch['code_ids'][i].unsqueeze(0).to(torch.device(args.device)),
                          batch['code_mask'][i].unsqueeze(0).to(torch.device(args.device))) for i in
                         range(1, batch_size)]

            positive_code_key = code_list.pop(0)
            inputs = {'input_ids': positive_code_key[0], 'attention_mask': positive_code_key[1]}
            positive_code_key = model(**inputs)[1]  # using pooled values

            negative_keys = []

            for code, mask in code_list:
                inputs = {'input_ids': code, 'attention_mask': mask}

                negative_key = model(**inputs)[1]  # using pooled values
                negative_keys.append(negative_key.clone().detach())

            negative_keys_reshaped = torch.cat(negative_keys, dim=0)

            loss = loss_formulation(query, positive_code_key, negative_keys_reshaped)
            loss.backward()

            all_losses.append(loss.to("cpu").detach().numpy())

            if (idx + 1) % args.num_of_accumulation_steps == 0:
                optimizer.zero_grad()
                optimizer.step()
            logging.debug(f"train_loss epoch {epoch}: {loss}")

        train_mean_loss = np.mean(all_losses)
        logging.info(f'Epoch {epoch} - Train-Loss: {train_mean_loss}')
    logging.info("Training finished")


def evaluation(args, model, test_params, valid_set):
    logging.info("Evaluate ...")
    print("Evaluate ...")
    eval_dataloader = DataLoader(valid_set, **test_params)
    all_distances = []
    for idx, batch in tqdm(enumerate(eval_dataloader)):

        if batch['code_ids'].size(0) < args.train_batch_size:
            logging.debug("continue")
            continue

        # query = doc
        query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': query_id, 'attention_mask': query_mask}
        query = model(**inputs)[1]  # using pooled values

        batch_size = args.eval_batch_size
        code_list = [(batch['code_ids'][i].unsqueeze(0).to(torch.device(args.device)),
                      batch['code_mask'][i].unsqueeze(0).to(torch.device(args.device))) for i in range(1, batch_size)]

        positive_code_key = code_list.pop(0)
        inputs = {'input_ids': positive_code_key[0], 'attention_mask': positive_code_key[1]}
        positive_code_key = model(**inputs)[1]  # using pooled values

        negative_keys = []

        for code, mask in code_list:
            inputs = {'input_ids': code, 'attention_mask': mask}

            negative_key = model(**inputs)[1]  # using pooled values
            negative_keys.append(negative_key.clone().detach())

        negative_keys_reshaped = torch.cat(negative_keys, dim=0)

        # calc Euclidean distance for positive key at first position
        distances = [np.linalg.norm((query - positive_code_key).detach().cpu().numpy())]

        for i in range(len(negative_keys_reshaped)):
            # calc Euclidean distance for negative keys
            distance = np.linalg.norm((query - negative_keys_reshaped[i]).detach().cpu().numpy())

            distances.append(distance)

        logging.debug("distances: %s", distances)
        all_distances.append(distances)

    logging.info("***** finished evaluation *****")
    return all_distances


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss_function", default="INFO_NCE", type=str, required=False,
                        help="Loss formulation selected in the list: " + ", ".join(LOSS_FUNCTIONS))

    parser.add_argument("--learning_architecture", default=None, type=str, required=False,
                        help="Learning architecture selected in the list: " + ", ".join(LEARNING_ARCHITECTURES))

    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-2,
                        help="For distributed training: local_rank")

    parser.add_argument("--log_path", default='../logging', type=str, required=False,
                        help="Path to log files")

    parser.add_argument("--lang", default='ruby', type=str, required=False, help="Language of the code")

    parser.add_argument("--train_batch_size", default=7, type=int, required=False, help="Training batch size")

    parser.add_argument("--eval_batch_size", default=7, type=int, required=False, help="Evaluation batch size")

    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False, help="Learning rate")

    parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="Number of training epochs")

    parser.add_argument("--n_labels", default=20, type=int, required=False, help="Number of labels")

    parser.add_argument("--train_size", default=0.8, type=float, required=False, help="percentage of train dataset used"
                                                                                      "for training")

    parser.add_argument("--mrr_path", default='../data/MRR.txt', type=str, required=False, help="Path to mrr file")

    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    parser.add_argument("--num_of_accumulation_steps", default=10, type=int, required=False,
                        help="Number of accumulation steps")

    parser.add_argument("--GPU", required=False, help="specify the GPU which should be used")

    args = parser.parse_args()
    args.dataset = 'codebert-base'
    args.model_name = 'microsoft/codebert-base'
    args.MAX_LEN = 512

    # Setup CUDA, GPU
    if args.GPU:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(filename=args.log_path + '/log_' + args.lang + '.txt',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=args.log_level)
    logging.info("Process rank: %s, device: %s, n_gpu: %s, language: %s, loss_formulation: %s",
                 args.local_rank, device, args.n_gpu, args.lang, args.loss_function)

    print("loglevel: ", args.log_level)

    # Set seed
    set_seed(args)

    # set up data data
    train_df, test_df = load_data(args)

    train_params = {'batch_size': args.train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': args.eval_batch_size,
                   'shuffle': True,
                   'num_workers': 0
                   }

    train_dataset = train_df.sample(frac=args.train_size, random_state=200)
    valid_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_df.reset_index(drop=True)

    logging.debug("TRAIN Dataset: %s", train_dataset.shape)
    logging.debug("VAL Dataset: %s", valid_dataset.shape)
    logging.debug("TEST Dataset: %s", test_dataset.shape)

    training_set = CustomDataset(train_dataset, args)
    test_set = CustomDataset(test_dataset, args)
    valid_set = CustomDataset(valid_dataset, args)

    # model

    model = RobertaModel.from_pretrained('microsoft/codebert-base')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    model.to(torch.device(args.device))

    if args.loss_function == 'triplet':
        loss_formulation = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    elif args.loss_function == 'INFO_NCE':
        loss_formulation = InfoNCE(negative_mode='unpaired')
    elif args.loss_function == 'Soft_nearest_neighbour':
        loss_formulation = torch.nn.SoftMarginLoss()
    else:
        raise ValueError("Loss formulation selected is not valid. Please select one of the following: " +
                         ", ".join(LOSS_FUNCTIONS))

    train(args, loss_formulation, model, optimizer, training_set)

    distances = evaluation(args, model, test_params, test_set)

    mrr = calculate_mrr_from_distances(distances)

    logging.info(f"MRR: {mrr}")

    # write mrr to file
    write_mrr_to_file(args, mrr)
    end_time = time.time()

    # Calculate runtime duration in seconds
    runtime_seconds = end_time - start_time

    # Convert runtime duration to hours, minutes, and seconds
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print or log the runtime
    logging.info(f"Program runtime: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


if __name__ == '__main__':
    main()
