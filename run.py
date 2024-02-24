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

LOSS_FUNCTIONS = ['triplet', 'INFO_NCE', 'soft_nearest_neighbour']
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
            truncation='longest_first',
            padding='max_length',
            return_token_type_ids=False
        )
        doc_ids = doc_inputs['input_ids']
        doc_mask = doc_inputs['attention_mask']

        code_inputs = self.tokenizer.encode_plus(
            code,
            add_special_tokens=False,
            max_length=self.max_len,
            truncation='longest_first',
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


def info_nce_loss(query, positive_key, negative_keys):
    # query = torch.flatten(query).view(-1).unsqueeze(0)
    # positive_key = torch.flatten(positive_key).view(-1).unsqueeze(0)
    # negative_keys = torch.stack([torch.flatten(code_emb).view(-1) for code_emb in negative_keys])
    return InfoNCE(negative_mode='unpaired')(query, positive_key, negative_keys)


def triplet_loss(query, positive_key, negative_key):
    # query = torch.flatten(query).view(-1).unsqueeze(0)
    # positive_key = torch.flatten(positive_key).view(-1).unsqueeze(0)
    # negative_key = torch.flatten(negative_key).view(-1).unsqueeze(0)
    return torch.nn.TripletMarginLoss(margin=1.0, p=2)(query, positive_key, negative_key)


def soft_nearest_neighbour_loss(query, positive_key, negative_keys):
    return torch.nn.SoftMarginLoss()(query, positive_key, negative_keys[0])


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


def train(args, model, optimizer, training_set):
    logging.info("Start training ...")
    print("Start training ...")
    for epoch in tqdm(range(1, args.num_train_epochs + 1)):
        logging.info("training epoch %d", epoch)

        # random.shuffle(training_set)
        train_dataloader = DataLoader(training_set, batch_size=args.train_batch_size, shuffle=True)

        all_losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", position=0, leave=True)
        for idx, batch in enumerate(progress_bar):

            if batch['code_ids'].size(0) < args.train_batch_size:
                logging.debug("continue")
                continue

            # query = doc
            query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': query_id, 'attention_mask': query_mask}
            query = model(**inputs)[1]  # using pooled values
            # query = model(**inputs)[0]  # using un-pooled values
            # print("query shape: ", query.shape)

            logging.debug(f"idx: {idx}")
            logging.debug(f"code_ids: {batch['code_ids'].shape}")
            logging.debug(f"code_ids_0: {batch['code_ids'][0].shape}")

            logging.debug(f"mask: {batch['code_mask'].shape}")
            logging.debug(f"mask_0: {batch['code_mask'][0].shape}")

            code_list = [(batch['code_ids'][i].unsqueeze(0).to(torch.device(args.device)),
                          batch['code_mask'][i].unsqueeze(0).to(torch.device(args.device))) for i in
                         range(0, args.train_batch_size)]

            # print("cl len: ",len(code_list))

            positive_code_key = code_list.pop(0)
            inputs = {'input_ids': positive_code_key[0], 'attention_mask': positive_code_key[1]}
            positive_code_key = model(**inputs)[1]  # using pooled values
            # positive_code_key = model(**inputs)[0]  # using un-pooled values

            negative_keys = []

            for code, mask in code_list:
                inputs = {'input_ids': code, 'attention_mask': mask}

                negative_key = model(**inputs)[1]  # using pooled values
                # negative_key = model(**inputs)[0]  # using un-pooled values
                negative_keys.append(negative_key.clone().detach())

            negative_keys_reshaped = torch.cat(negative_keys[:min(args.num_of_negative_samples, len(negative_keys))],
                                               dim=0)

            if args.loss_function == 'INFO_NCE':
                loss = info_nce_loss(query, positive_code_key, negative_keys_reshaped)
            elif args.loss_function == 'triplet':
                negative_key = negative_keys_reshaped[0].unsqueeze(0)
                loss = triplet_loss(query, positive_code_key, negative_key)
            else:
                loss = soft_nearest_neighbour_loss(query, positive_code_key, negative_keys_reshaped)

            # print(loss)
            loss.backward()

            all_losses.append(loss.to("cpu").detach().numpy())

            if (idx + 1) % args.num_of_accumulation_steps == 0:
                # print("model updated")
                optimizer.step()
                optimizer.zero_grad()
            logging.debug(f"train_los s epoch {epoch}: {loss}")

        train_mean_loss = np.mean(all_losses)
        logging.info(f'Epoch {epoch} - Train-Loss: {train_mean_loss}')

        del train_dataloader
    logging.info("Training finished")


def evaluation(args, model, valid_set):
    logging.info("Evaluate ...")
    print("Evaluate ...")

    batch_size = 1
    eval_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    all_distances = []
    progress_bar = tqdm(eval_dataloader, position=0, leave=True)
    for idx, batch in enumerate(progress_bar):

        query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': query_id, 'attention_mask': query_mask}
        query = model(**inputs)[1]  # using pooled values

        positive_code_key_id = batch['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        positive_code_key_mask = batch['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': positive_code_key_id, 'attention_mask': positive_code_key_mask}
        positive_code_key = model(**inputs)[1]  # using pooled values

        # negative_keys
        sample_indices = list(range(len(eval_dataloader)))
        sample_indices.remove(idx)
        sample_indices = random.choices(sample_indices, k=min(args.num_of_distractors, len(sample_indices)))

        subset = torch.utils.data.Subset(valid_set, sample_indices)
        data_loader_subset = DataLoader(subset, batch_size=batch_size, shuffle=True)

        negative_keys = []
        for idx2, batch2 in enumerate(data_loader_subset):
            code_id = batch2['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            code_mask = batch2['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': code_id, 'attention_mask': code_mask}
            negative_key = model(**inputs)[1]
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

    parser.add_argument("--log_path", default='../logging', type=str, required=False,
                        help="Path to log files")

    parser.add_argument("--lang", default='ruby', type=str, required=False, help="Language of the code")

    parser.add_argument("--train_batch_size", default=16, type=int, required=False, help="Training batch size")

    parser.add_argument("--eval_batch_size", default=16, type=int, required=False, help="Evaluation batch size")

    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False, help="Learning rate")

    parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="Number of training epochs")

    parser.add_argument("--train_size", default=0.8, type=float, required=False, help="percentage of train dataset used"
                                                                                      "for training")

    parser.add_argument("--mrr_path", default='./data/MRR.txt', type=str, required=False, help="Path to mrr file")

    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    parser.add_argument("--num_of_accumulation_steps", default=16, type=int, required=False,
                        help="Number of accumulation steps")

    parser.add_argument("--num_of_negative_samples", default=15, type=int, required=False, help="Number of negative "
                                                                                                "samples")
    parser.add_argument("--num_of_distractors", default=99, type=int, required=False, help="Number of distractors")
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
    logging.info("Device: %s, n_gpu: %s, language: %s, loss_formulation: %s",
                 device, args.n_gpu, args.lang, args.loss_function)

    logging.debug("args: %s", args)
    print("loglevel: ", args.log_level)


    # Set seed
    set_seed(args)

    # set up data data
    train_df, test_df = load_data(args)

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

    # train
    train(args, model, optimizer, training_set)

    # evaluate
    distances = evaluation(args, model, test_set)

    mrr = calculate_mrr_from_distances(distances)
    logging.info(f"MRR: {mrr}")

    # write mrr to file
    write_mrr_to_file(args, mrr)

    # Calculate runtime duration in seconds
    end_time = time.time()
    runtime_seconds = end_time - start_time

    # Convert runtime duration to hours, minutes, and seconds
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print or log the runtime
    print(f"Program runtime: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


if __name__ == '__main__':
    main()
