import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizer

import logging


class CustomDataset(TensorDataset):
    """
    Dataset for the project uses codebert-base Roberta Tokenizer
    contains input IDs and attention masks for a NL description and its corresponding Code
    """

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


def load_data(args):
    code_search_dataset = load_dataset('code_search_net', args.lang, trust_remote_code=True)

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


def load_stat_code_search_dataset(args):
    dataset_path = args.data_path + "datasets/test_statcodesearch.jsonl"

    file = open(dataset_path, "r", encoding="utf8")
    json_obj = pd.read_json(path_or_buf=file, lines=True)
    # creating custom dataset from the list file containing 3000 words
    function_code = []
    function_documentation = []
    for i in range(len(json_obj['input'])):
        code = json_obj["input"][i].split("[CODESPLIT]")[1]
        doc = json_obj["input"][i].split("[CODESPLIT]")[0]
        function_code.append(str(code))
        function_documentation.append(str(doc))

    test_df = pd.DataFrame()
    test_df['doc'] = function_documentation
    test_df['code'] = function_code

    return test_df


def calculate_cosine_distance(query, positive_code_key):
    # Compute cosine distance
    return cosine_distances(query, positive_code_key)


def calculate_mrr_from_distances(distances_lists):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a list of distances where the positive value is at the first index

    :param distances_lists:
    :return:
    """
    ranks = []
    for batch_idx, predictions in enumerate(distances_lists):
        correct_score = predictions[0]
        scores = np.array([prediction for prediction in predictions])
        rank = np.sum(scores <= correct_score)
        ranks.append(rank)
    mean_mrr = np.mean(1.0 / np.array(ranks))

    return mean_mrr


def write_mrr_to_file(args, mrr, gen_mrr, runtime=" ", test=False, generalisation=False):
    """
    Custom MRR and hyperparameter logging

    :param args:
    :param mrr:
    :param gen_mrr: generalization MRR
    :param runtime:
    :param test: True for personal test, False for actually running the program
    :param generalisation: True when doing the Generaliation task (train on  Go/python/... and test on R)
    """
    mrr_path = args.data_path + "MRR.txt"
    with open(mrr_path, "r") as file:
        mrr_old = file.read()

    with open(mrr_path, "w") as file:
        now = datetime.now().strftime("%d/%m/%Y%H:%M")

        mrr_header = ""

        if test:
            mrr_header += "(test)"

        if generalisation:
            mrr_header += "Generalisation:"

        mrr_header += (
            f"{now}: {args.lang} {args.loss_function} {args.architecture} epochs:{args.num_train_epochs} batch_size:{args.batch_size} "
            f"learning_rate:{args.learning_rate} acccumulation_steps:{args.num_of_accumulation_steps} "
            f"distractors:{args.num_of_distractors} runtime:{runtime} MRR:{mrr} general_MRR: {gen_mrr}\n")

        mrr_new = f"{mrr_old}{mrr_header}"
        file.write(mrr_new)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def visualize_losses(train_losses, val_losses, args):
    """
    Visualize training and validation losses

    :param train_losses:
    :param val_losses:
    :param args:
    """

    logging.info("Visualize losses")

    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # save plot
    filepath = args.data_path + f"plots/{args.lang}/{args.architecture}_{args.loss_function}_losses.png"
    plt.savefig(filepath)
    # plt.show()


def visualize_embeddings(args, idx, query_embedding, positive_embedding, negative_embeddings, first_time):
    # Combine all embeddings
    # all_embeddings = [query_embedding, positive_embedding] + [negative_embeddings]

    all_embeddings = np.concatenate((query_embedding, positive_embedding, negative_embeddings), axis=0)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plot the embeddings
    plt.figure(figsize=(10, 8))

    # Plot query embedding
    plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='blue', label='Query', zorder=2, marker='o')

    # Plot positive embedding
    plt.scatter(embeddings_2d[1, 0], embeddings_2d[1, 1], color='green', label='Positive', zorder=2, marker='o')

    # Plot negative embeddings
    for i in range(2, len(all_embeddings)):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color='red', label='Negative', zorder=1, marker='x')

    plt.legend(['Query', 'Positive', 'Negative'])
    plt.title('t-SNE Visualization of Embeddings')

    # Save plot
    filename = f"plots/{args.lang}/{args.architecture}_{args.loss_function}_{args.lang}_{idx}"
    if first_time:
        filepath = args.data_path + filename + ".png"
    else:
        filepath = args.data_path + filename + "_after_training.png"
    plt.savefig(filepath)
    # plt.show()


def visualize(args, model, visualization_set, first_time=True):
    """
    this methode plots the first three entrys in the visualisation dataset.


    :param args:
    :param model:
    :param visualization_set:
    :param first_time: specifys the name of the files the plots will be saved in
    :return:
    """
    logging.info("Visualize ...")

    batch_size = 1
    visual_dataloader = DataLoader(visualization_set, batch_size=batch_size, shuffle=False)

    for idx, batch in enumerate(visual_dataloader):

        if idx > 2:
            break

        negative_keys_reshaped, positive_code_key, query = compute_embeddings_visualization(args, batch, batch_size, idx,
                                                                                            model, visual_dataloader,
                                                                                            visualization_set)

        visualize_embeddings(args, idx, query.detach().cpu().numpy(), positive_code_key.detach().cpu().numpy(),
                             negative_keys_reshaped.detach().cpu().numpy(), first_time)


def compute_embeddings_visualization(args, batch, batch_size, idx, model, visual_dataloader, visualization_set):
    """
    2 Versions to retrieve positive and negative keys from the model for the visualization
    One for MoCo and the other one for Uni- and Bi-encoder setups

    :param args:
    :param batch:
    :param batch_size:
    :param idx:
    :param model:
    :param visual_dataloader:
    :param visualization_set:
    :return:
    """
    # query
    query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
    query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
    inputs = {'input_ids': query_id, 'attention_mask': query_mask}
    query = model_call(args, model, inputs, False)

    # keys

    # setup dataloader for negative_keys
    sample_indices = list(range(len(visual_dataloader)))
    sample_indices.remove(idx)
    sample_indices = random.choices(sample_indices, k=min(args.num_of_distractors, len(sample_indices)))
    subset = torch.utils.data.Subset(visualization_set, sample_indices)
    data_loader_subset = DataLoader(subset, batch_size=batch_size, shuffle=True)
    progress_bar = tqdm(data_loader_subset, desc="encode negatives", position=1, leave=True, dynamic_ncols=True)

    if args.architecture == "MoCo":

        # encode and add negative keys to queue
        for idx2, batch2 in enumerate(progress_bar):
            code_id = batch2['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            code_mask = batch2['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': code_id, 'attention_mask': code_mask}
            model(True, **inputs)

        # retrieve keys
        # positive key
        positive_code_key_id = batch['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        positive_code_key_mask = batch['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': positive_code_key_id, 'attention_mask': positive_code_key_mask}

        positive_code_key, negative_code_keys = model_call(args, model, inputs, True, visualization=True)

        negative_code_keys_reshaped = torch.cat(negative_code_keys)

        return negative_code_keys_reshaped, positive_code_key, query

    else:

        # positive keys
        positive_code_key_id = batch['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        positive_code_key_mask = batch['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': positive_code_key_id, 'attention_mask': positive_code_key_mask}
        positive_code_key = model_call(args, model, inputs, True)

        negative_keys = []
        for idx2, batch2 in enumerate(progress_bar):
            code_id = batch2['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            code_mask = batch2['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': code_id, 'attention_mask': code_mask}
            negative_key = model_call(args, model, inputs, True)
            negative_keys.append(negative_key.clone().detach())

        negative_keys_reshaped = torch.cat(negative_keys)

        return negative_keys_reshaped, positive_code_key, query


def compute_embeddings_train(args, batch, model, validation=False):
    """
    Two versions to retrieve output from the model for query and negative & positive examples
    One for MoCo and the other one for Uni- and Bi-encoder setups

    :param args:
    :param batch:
    :param model:
    :param validation:
    :return:
    """

    # query = doc
    query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
    query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
    inputs = {'input_ids': query_id, 'attention_mask': query_mask}
    query = model_call(args, model, inputs, False)

    # Logging ***************

    logging.debug(f"query_id: {query_id.shape}")
    logging.debug(f"query_mask: {query_mask.shape}")
    logging.debug(f"query shape model output: {query.shape}")
    logging.debug("__Shapes for Code keys:___________")
    logging.debug(f"code_ids: {batch['code_ids'].shape}")
    logging.debug(f"code_ids_0: {batch['code_ids'][0].shape}")
    logging.debug(f"code_mask: {batch['code_mask'].shape}")
    logging.debug(f"code_mask_0: {batch['code_mask'][0].shape}")

    # ***************

    code_list = [(batch['code_ids'][i].unsqueeze(0).to(torch.device(args.device)),
                  batch['code_mask'][i].unsqueeze(0).to(torch.device(args.device))) for i in
                 range(0, args.batch_size)]

    if args.architecture == "MoCo":

        for i in range(1, len(code_list)):  # negatives
            code_id = batch['code_ids'][i].unsqueeze(0).to(torch.device(args.device))
            code_mask = batch['code_mask'][i].unsqueeze(0).to(torch.device(args.device))
            inputs = {'input_ids': code_id, 'attention_mask': code_mask}

            model(True, **inputs)  # encode negatives and add them to the queue

        # positive key
        code_id = batch['code_ids'][0].unsqueeze(0).to(torch.device(args.device))
        code_mask = batch['code_mask'][0].unsqueeze(0).to(torch.device(args.device))
        inputs = {'input_ids': code_id, 'attention_mask': code_mask}

        positive_code_key, negative_code_keys = model_call(args, model, inputs, True, validation=validation)

        negative_code_keys_reshaped = torch.cat(negative_code_keys)

        return negative_code_keys_reshaped, positive_code_key, query

    else:

        # keys (codes)
        positive_code_key = code_list.pop(0)
        inputs = {'input_ids': positive_code_key[0], 'attention_mask': positive_code_key[1]}
        positive_code_key = model_call(args, model, inputs, True)

        negative_keys = []
        for code, mask in code_list:
            inputs = {'input_ids': code, 'attention_mask': mask}
            negative_code_key = model_call(args, model, inputs, True)
            negative_keys.append(negative_code_key.clone().detach())

        negative_keys_reshaped = torch.cat(negative_keys[:min(args.num_of_negative_samples, len(negative_keys))], dim=0)

        logging.debug("keys model output:")
        logging.debug(f"positive key: {positive_code_key.shape}")
        logging.debug(f"negative key0: {negative_keys[0].shape}, reshaped: {negative_keys_reshaped.shape}")

        return negative_keys_reshaped, positive_code_key, query


def model_call(args, model, model_input, is_code_key: bool, visualization: bool = False, validation: bool = False):
    """
    This is the function that does the actual model call.

    For the Uni and Bi architecture if is_code_cey = True each positive and negative keys are requested separately but
    for MoCo positive and negative Keys are returned at the same time.

    :param args: The program arguments
    :param model: The Model that is currently trained/evaluated
    :param model_input: a dictionary with the input IDs at the first position and the Masks at the second position
    :param is_code_key: True if its a code key or False if its a NL query
    :param visualization: True if the program is currently visualizing data, False if not
    :param validation: True if the program is currently validating, False if not
    :return: the Model output
    """
    if args.architecture == "Uni":
        output = model(**model_input)
    elif args.architecture == "Bi":
        output = model(is_code_key, **model_input)
    elif args.architecture == "MoCo":
        if is_code_key:
            positive_key, negative_keys = model(is_code_key, **model_input, visualization=visualization, validation=validation)
            return positive_key, negative_keys
        else:
            output = model(is_code_key, **model_input)
    else:
        raise ValueError(f'Architecture {args.architecture} not supported')

    return output
