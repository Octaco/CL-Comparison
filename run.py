import argparse
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from info_nce import InfoNCE
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizer

import logging
from models import UniEncoderModel, BiEncoderModel, MoCoModel

LOSS_FUNCTIONS = ['triplet', 'InfoNCE', 'ContrastiveLoss']
ARCHITECTURES = ['Uni', 'Bi', 'MoCo']


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
    return InfoNCE(negative_mode='unpaired')(query, positive_key, negative_keys)


def triplet_loss(query, positive_key, negative_key):
    return torch.nn.TripletMarginLoss(margin=1.0, p=2)(query, positive_key, negative_key)


def contrastive_loss(args, query, key, label):
    # Prepare labels
    query = query.to(args.device)
    # Adjust dimensions of the key
    key = key.squeeze().unsqueeze(0).to(args.device)

    # Prepare target tensor
    if label == 1:
        target = torch.tensor([1]).to(args.device)
    elif label == -1:
        target = torch.tensor([-1]).to(args.device)
    else:
        raise ValueError('this should not happen')
    # Calculate loss
    loss_func = torch.nn.CosineEmbeddingLoss()
    loss = loss_func(query, key, target)

    return loss


def calculate_cosine_distance(query, positive_code_key):
    # Compute cosine distance
    return cosine_distances(query, positive_code_key)


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


def model_call(args, model, model_input, is_code_key, visualization=False):
    if args.architecture == "Uni":
        output = model(**model_input)
    elif args.architecture == "Bi":
        output = model(is_code_key, **model_input)
    elif args.architecture == "MoCo":
        if is_code_key:
            positive_key, negative_keys = model(is_code_key, **model_input, visualization=visualization)
            return positive_key, negative_keys
        else:
            output = model(is_code_key, **model_input)
    else:
        raise ValueError(f'Architecture {args.architecture} not supported')

    return output


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


def write_mrr_to_file(args, mrr, gen_mrr, runtime=" ", test=False, generalisation=False):
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
            f"{now}: {args.lang} {args.loss_function} {args.architecture}epochs:{args.num_train_epochs} batch_size:{args.train_batch_size} "
            f"learning_rate:{args.learning_rate} acccumulation_steps:{args.num_of_accumulation_steps} "
            f"distractors:{args.num_of_distractors} runtime:{runtime} MRR:{mrr} general_MRR: {gen_mrr}\n")

        mrr_new = f"{mrr_old}{mrr_header}"
        file.write(mrr_new)


def visualize_losses(train_losses, val_losses, args):
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
    plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='blue', label='Query', zorder=2)

    # Plot positive embedding
    plt.scatter(embeddings_2d[1, 0], embeddings_2d[1, 1], color='green', label='Positive', zorder=2)

    # Plot negative embeddings
    for i in range(2, len(all_embeddings)):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color='red', label='Negative', zorder=1)

    plt.legend(['Query', 'Positive', 'Negative'])

    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Save plot
    filename = f"plots/{args.lang}/{args.architecture}_{args.loss_function}_embeddings{idx}"
    if first_time:
        filepath = args.data_path + filename + ".png"
    else:
        filepath = args.data_path + filename + "_after_training.png"
    plt.savefig(filepath)
    # plt.show()


def visualize(args, model, visualization_set, first_time=True):
    logging.info("Visualize ...")

    batch_size = 1
    visual_dataloader = DataLoader(visualization_set, batch_size=batch_size, shuffle=False)

    for idx, batch in enumerate(visual_dataloader):

        if idx > 2:
            break

        negative_keys_reshaped, positive_code_key, query = get_model_output_visualization(args, batch, batch_size, idx,
                                                                                          model, visual_dataloader,
                                                                                          visualization_set)

        visualize_embeddings(args, idx, query.detach().cpu().numpy(), positive_code_key.detach().cpu().numpy(),
                             negative_keys_reshaped.detach().cpu().numpy(), first_time)


def train(args, model, optimizer, training_set, valid_set):
    logging.info("Start training ...")
    all_train_mean_losses = []
    all_val_mean_losses = []
    for epoch in tqdm(range(1, args.num_train_epochs + 1), desc="Epochs", dynamic_ncols=True, position=0, leave=True):
        logging.info("training epoch %d", epoch)

        # random.shuffle(training_set)
        train_dataloader = DataLoader(training_set, batch_size=args.train_batch_size, shuffle=True)

        all_losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Train", position=1, leave=True, dynamic_ncols=True)
        for idx, batch in enumerate(progress_bar):

            logging.debug("____________________________________________________________")
            logging.debug(f"idx: {idx}")
            if batch['code_ids'].size(0) < args.train_batch_size:
                logging.debug("continue")
                continue

            negative_keys_reshaped, positive_code_key, query = get_model_output_training(args, batch, model)

            if args.loss_function == 'InfoNCE':
                loss = info_nce_loss(query, positive_code_key, negative_keys_reshaped)
            elif args.loss_function == 'triplet':
                negative_key = negative_keys_reshaped[0].unsqueeze(0)
                loss = triplet_loss(query, positive_code_key, negative_key)
            elif args.loss_function == 'ContrastiveLoss':
                loss = contrastive_loss(args, query, positive_code_key, 1)
                all_losses.append(loss.to("cpu").detach().numpy())
                loss.backward(retain_graph=True)
                loss = contrastive_loss(args, query, negative_keys_reshaped[0], -1)

            else:
                exit("Loss function not supported")

            all_losses.append(loss.to("cpu").detach().numpy())
            loss.backward()

            if (idx + 1) % args.num_of_accumulation_steps == 0:
                # print("model updated")
                optimizer.step()
                optimizer.zero_grad()
            logging.debug(f"train_loss epoch {epoch}: {loss}")

        train_mean_loss = np.mean(all_losses)
        all_train_mean_losses.append(train_mean_loss)
        logging.info(f'Epoch {epoch} - Train-Loss: {train_mean_loss}')

        ########################################################
        # validation
        ########################################################

        validation_dataloader = DataLoader(valid_set, batch_size=args.train_batch_size, shuffle=True)
        all_val_losses = []
        progress_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch} eval ", position=2, leave=True,
                            dynamic_ncols=True)
        for idx, batch in enumerate(progress_bar):
            if batch['code_ids'].size(0) < args.train_batch_size:
                logging.debug("continue")
                continue

            negative_keys_reshaped, positive_code_key, query = get_model_output_training(args, batch, model)

            if args.loss_function == 'InfoNCE':
                loss = info_nce_loss(query, positive_code_key, negative_keys_reshaped)
            elif args.loss_function == 'triplet':
                negative_key = negative_keys_reshaped[0].unsqueeze(0)
                loss = triplet_loss(query, positive_code_key, negative_key)
            elif args.loss_function == 'ContrastiveLoss':
                # compute loss for positive key
                loss = contrastive_loss(args, query, positive_code_key, 1)
                all_val_losses.append(loss.to("cpu").detach().numpy())
                # compute loss for negative key
                loss = contrastive_loss(args, query, negative_keys_reshaped[0], -1)
            else:
                exit("Loss function not supported")

            all_val_losses.append(loss.to("cpu").detach().numpy())
        val_mean_loss = np.mean(all_val_losses)
        all_val_mean_losses.append(val_mean_loss)
        logging.info(f'Epoch {epoch} - val-Loss: {val_mean_loss}')

        del validation_dataloader

    logging.info("Training finished")
    return all_train_mean_losses, all_val_mean_losses


def predict_distances(args, model, test_set):
    logging.info("Start Evaluation...")
    batch_size = 1
    num_distractors = args.num_of_distractors

    device = args.device
    eval_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if args.architecture == "MoCo":

        progress_bar = tqdm(eval_dataloader, desc="Evaluation")

        all_distances = []
        for idx, batch in enumerate(progress_bar):

            # Sample Query
            query_id = batch['doc_ids'][0].to(torch.device(device)).unsqueeze(0)
            query_mask = batch['doc_mask'][0].to(torch.device(device)).unsqueeze(0)
            inputs = {'input_ids': query_id, 'attention_mask': query_mask}
            query = model_call(args, model, inputs, False)  # CodeBERT pooled
            query = query.detach().cpu().numpy()

            if idx < 1:
                # setup dataloader for negative_keys, first time
                sample_indices = list(range(len(eval_dataloader)))
                sample_indices.remove(idx)
                sample_indices = random.choices(sample_indices, k=min(args.num_of_distractors, len(sample_indices)))
                subset = torch.utils.data.Subset(test_set, sample_indices)
                data_loader_subset = DataLoader(subset, batch_size=batch_size, shuffle=True)

                # encode and add negative keys to queue
                for idx2, batch2 in enumerate(data_loader_subset):
                    code_id = batch2['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
                    code_mask = batch2['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
                    inputs = {'input_ids': code_id, 'attention_mask': code_mask}
                    model(True, **inputs)

            # sample keys
            positive_code_key_id = batch['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            positive_code_key_mask = batch['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': positive_code_key_id, 'attention_mask': positive_code_key_mask}

            positive_code_key, negative_code_keys = model_call(args, model, inputs, True, visualization=True)

            negative_code_keys = np.array(negative_code_keys.cpu())

            # calculate distances
            distances = [calculate_cosine_distance(query, positive_code_key)]

            for key in negative_code_keys:
                distance = calculate_cosine_distance(query, key)
                distances.append(distance)

            all_distances.append(distances)

        logging.info("finished Evaluation")
        return all_distances

    else:
        # Compute code embeddings
        code_embs = []
        progress_bar = tqdm(eval_dataloader, desc="Create Code Embeddings")
        for idx, batch in enumerate(progress_bar):
            code_id = batch['code_ids'][0].to(torch.device(device)).unsqueeze(0)
            code_mask = batch['code_mask'][0].to(torch.device(device)).unsqueeze(0)
            inputs = {'input_ids': code_id, 'attention_mask': code_mask}
            code_emb = model_call(args, model, inputs, True)  # CodeBERT pooled
            code_embs.append(code_emb.detach().cpu().numpy())

        code_embs = np.array(code_embs)
        # print(code_embs.shape)

        all_distances = []
        progress_bar = tqdm(eval_dataloader, desc="Compute Distances")
        for idx, batch in enumerate(progress_bar):

            # Sample Query
            query_id = batch['doc_ids'][0].to(torch.device(device)).unsqueeze(0)
            query_mask = batch['doc_mask'][0].to(torch.device(device)).unsqueeze(0)
            inputs = {'input_ids': query_id, 'attention_mask': query_mask}
            query = model_call(args, model, inputs, False)  # CodeBERT pooled
            query = query.detach().cpu().numpy()
            # print("query shape :", query.shape)

            # Sample Positive Code
            positive_code = code_embs[idx]

            # print("positive shape :", positive_code.shape)

            # calc Cosine distance for positive code
            distances = [calculate_cosine_distance(query, positive_code)]

            # Sample Negative Codes
            sample_indices = list(range(len(code_embs)))
            sample_indices.remove(idx)
            sample_indices = random.choices(code_embs, k=min(num_distractors, len(sample_indices)))

            negative_codes = []
            for idx in range(len(sample_indices)):
                negative_codes.append(code_embs[idx])
            negative_codes = np.array(negative_codes)
            # print("negatve shape :", negative_codes.shape)

            for i in range(len(negative_codes)):
                # calc cosine distance for negative keys
                distance = calculate_cosine_distance(query, negative_codes[i])
                distances.append(distance)

            all_distances.append(distances)

        logging.info("finished Evaluation")

        return all_distances


def get_model_output_visualization(args, batch, batch_size, idx, model, visual_dataloader, visualization_set):
    # query
    query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
    query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
    inputs = {'input_ids': query_id, 'attention_mask': query_mask}
    query = model_call(args, model, inputs, False)

    #keys

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


def get_model_output_training(args, batch, model):
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
                 range(0, args.train_batch_size)]

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

        positive_code_key, negative_code_keys = model_call(args, model, inputs, True)

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


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss_function", default="InfoNCE", type=str, required=False,
                        help="Loss formulation selected in the list: " + ", ".join(LOSS_FUNCTIONS))

    parser.add_argument("--architecture", default="Uni", type=str, required=False,
                        help="Learning architecture selected in the list: " + ", ".join(ARCHITECTURES))

    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--log_path", default='./logging', type=str, required=False,
                        help="Path to log files")

    parser.add_argument("--lang", default='ruby', type=str, required=False, help="Language of the code")

    parser.add_argument("--train_batch_size", default=16, type=int, required=False, help="Training batch size")

    parser.add_argument("--eval_batch_size", default=16, type=int, required=False, help="Evaluation batch size")

    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False, help="Learning rate")

    parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="Number of training epochs")

    parser.add_argument("--momentum", default=0.999, required=False, help="Momentum parameter")

    parser.add_argument("--train_size", default=0.8, type=float, required=False, help="percentage of train dataset used"
                                                                                      "for training")

    parser.add_argument("--data_path", default='./data/', type=str, required=False, help="Path to data folder")

    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    parser.add_argument("--num_of_accumulation_steps", default=16, type=int, required=False,
                        help="Number of accumulation steps")

    parser.add_argument("--num_of_negative_samples", default=15, type=int, required=False, help="Number of negative "
                                                                                                "samples")
    parser.add_argument("--num_of_distractors", default=99, type=int, required=False, help="Number of distractors")
    parser.add_argument("--GPU", required=False, help="specify the GPU which should be used")
    parser.add_argument("--do_generalisation", default=True, type=bool, required=False)

    args = parser.parse_args()
    args.dataset = 'codebert-base'
    args.model_name = 'microsoft/codebert-base'
    args.MAX_LEN = 256

    # set mini-batchsize to 2 for triplet and contrastive loss
    # and macro-batchsize to (num_accumulation_steps * train-batchsize) / 2
    # and set num of negatives for triplet and ContrastiveLoss
    if args.loss_function == "triplet" or args.loss_function == "ContrastiveLoss":
        args.num_of_accumulation_steps = (args.num_of_accumulation_steps * args.train_batch_size) / 2
        args.train_batch_size = 2
        args.eval_batch_size = 2
        args.num_of_negative_samples = 1

    # Setup CUDA, GPU
    if args.GPU:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    filename = args.log_path + '/log_' + args.lang + '_' + args.architecture + '_' + args.loss_function + '.txt'
    logging.basicConfig(filename=filename,
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
    visualization_dataset = test_dataset.sample(args.num_of_distractors)
    visualization_dataset = visualization_dataset.reset_index(drop=True)

    logging.debug("TRAIN Dataset: %s", train_dataset.shape)
    logging.debug("VAL Dataset: %s", valid_dataset.shape)
    logging.debug("TEST Dataset: %s", test_dataset.shape)

    training_set = CustomDataset(train_dataset, args)
    test_set = CustomDataset(test_dataset, args)
    valid_set = CustomDataset(valid_dataset, args)
    visualization_set = CustomDataset(visualization_dataset, args)

    # model

    if args.architecture == 'Uni':
        model = UniEncoderModel(args.model_name)
    elif args.architecture == 'Bi':
        model = BiEncoderModel(args.model_name)
    elif args.architecture == 'MoCo':
        model = MoCoModel(args)
    else:
        exit("Learning architecture not supported")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    model.to(torch.device(args.device))

    # visualize
    visualize(args, model, visualization_set, True)

    # train
    train_losses, val_losses = train(args, model, optimizer, training_set, valid_set)

    # visualize train and val losses
    visualize_losses(train_losses, val_losses, args)

    # visualize again
    visualize(args, model, visualization_set, False)

    # evaluate
    distances = predict_distances(args, model, test_set)

    mrr = calculate_mrr_from_distances(distances)
    logging.info(f"MRR: {mrr}")

    #####################
    # test generalisation
    #####################
    generalisation_mrr = 0
    if args.do_generalisation:
        logging.info("Start Generalisation...")
        generalisation_df = load_stat_code_search_dataset(args)

        generalisation_set = CustomDataset(generalisation_df, args)
        distances = predict_distances(args, model, generalisation_set)

        generalisation_mrr = calculate_mrr_from_distances(distances)
        logging.info(f"Generalisation MRR: {generalisation_mrr}")

    # Calculate runtime duration in seconds
    end_time = time.time()
    runtime_seconds = end_time - start_time

    # Convert runtime duration to hours, minutes, and seconds
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    #  log the runtime
    runtime = f"{int(hours)}:{int(minutes)}:{int(seconds)}"
    logging.info(f"Program runtime: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

    # write mrr to file
    write_mrr_to_file(args, mrr, generalisation_mrr, runtime)


if __name__ == '__main__':
    main()
