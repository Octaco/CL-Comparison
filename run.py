import argparse
from sklearn.manifold import TSNE

import logging
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from info_nce import InfoNCE
from transformers import RobertaTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

from datetime import datetime
import time
from tqdm import tqdm
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


def model_call(args, model, model_input, is_code_key):

    if args.architecture == "Uni":
        output = model(**model_input)
    elif args.architecture == "Bi":
        output = model(is_code_key, **model_input)
    elif args.architecture == "MoCo":
        output = model(**model_input)
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


def write_mrr_to_file(args, mrr, runtime=" ", test=False):
    mrr_path = args.data_path + "MRR.txt"
    with open(mrr_path, "r") as file:
        mrr_old = file.read()

    with open(mrr_path, "w") as file:
        now = datetime.now().strftime("%d/%m/%Y%H:%M")

        if test:
            mrr_addition = "(test)"
        else:
            mrr_addition = ""

        mrr_addition += (
            f"{now}: {args.lang} {args.loss_function} {args.architecture}epochs:{args.num_train_epochs} batch_size:{args.train_batch_size} "
            f"learning_rate:{args.learning_rate} acccumulation_steps:{args.num_of_accumulation_steps} "
            f"distractors:{args.num_of_distractors} runtime:{runtime} MRR:{mrr}\n")

        mrr_new = f"{mrr_old}{mrr_addition}"
        file.write(mrr_new)


def visualize_losses(train_losses, val_losses, args):
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # save plot
    filepath = args.data_path + "/plots/losses.png"
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
    plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='blue', label='Query')

    # Plot positive embedding
    plt.scatter(embeddings_2d[1, 0], embeddings_2d[1, 1], color='green', label='Positive')

    # Plot negative embeddings
    for i in range(2, len(all_embeddings)):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color='red', label='Negative')

    plt.legend(['Query', 'Positive', 'Negative'])

    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Save plot
    if first_time:
        filepath = args.data_path + f"plots/embeddings{idx}.png"
    else:
        filepath = args.data_path + f"plots/embeddings{idx}_after_training.png"
    plt.savefig(filepath)
    # plt.show()


def visualize(args, model, visualisation_set, first_time=True):
    logging.info("Visualize ...")

    batch_size = 1
    visual_dataloader = DataLoader(visualisation_set, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    for idx, batch in enumerate(visual_dataloader):

        if idx > 2:
            break

        query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': query_id, 'attention_mask': query_mask}
        query = model_call(args, model, inputs, False)

        positive_code_key_id = batch['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        positive_code_key_mask = batch['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': positive_code_key_id, 'attention_mask': positive_code_key_mask}
        positive_code_key = model_call(args, model, inputs, True)

        # negative_keys
        sample_indices = list(range(len(visual_dataloader)))
        sample_indices.remove(idx)
        sample_indices = random.choices(sample_indices, k=min(args.num_of_distractors, len(sample_indices)))

        subset = torch.utils.data.Subset(visualisation_set, sample_indices)
        data_loader_subset = DataLoader(subset, batch_size=batch_size, shuffle=True)

        negative_keys = []
        for idx2, batch2 in enumerate(data_loader_subset):
            code_id = batch2['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            code_mask = batch2['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': code_id, 'attention_mask': code_mask}
            negative_key = model_call(args, model, inputs, True)
            negative_keys.append(negative_key.clone().detach())

        negative_keys_reshaped = torch.cat(negative_keys, dim=0)

        all_embeddings.append((query, positive_code_key, negative_keys))
        visualize_embeddings(args, idx, query.detach().cpu().numpy(), positive_code_key.detach().cpu().numpy(),
                             negative_keys_reshaped.detach().cpu().numpy(), first_time)


def train(args, model, optimizer, training_set, valid_set):
    logging.info("Start training ...")
    print("Start training ...")
    all_train_mean_losses = []
    all_val_mean_losses = []
    for epoch in tqdm(range(1, args.num_train_epochs + 1), desc="Epochs", dynamic_ncols=True, position=0, leave=True):
        logging.info("training epoch %d", epoch)

        # random.shuffle(training_set)
        train_dataloader = DataLoader(training_set, batch_size=args.train_batch_size, shuffle=True)

        all_losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Train", position=1, leave=True, dynamic_ncols=True)
        for idx, batch in enumerate(progress_bar):

            if batch['code_ids'].size(0) < args.train_batch_size:
                logging.debug("continue")
                continue

            # query = doc
            query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': query_id, 'attention_mask': query_mask}
            query = model_call(args, model, inputs, False)
            # query = model(**inputs)[1]  # using pooled values
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
            positive_code_key = model_call(args, model, inputs, True)
            # positive_code_key = model(**inputs)[1]  # using pooled values
            # positive_code_key = model(**inputs)[0]  # using un-pooled values

            negative_keys = []
            for code, mask in code_list:
                inputs = {'input_ids': code, 'attention_mask': mask}

                negative_code_key = model_call(args, model, inputs, True)
                # negative_key = model(**inputs)[1]  # using pooled values
                # negative_key = model(**inputs)[0]  # using un-pooled values
                negative_keys.append(negative_code_key.clone().detach())

            negative_keys_reshaped = torch.cat(negative_keys[:min(args.num_of_negative_samples, len(negative_keys))],
                                               dim=0)

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
            # print(loss)
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

            query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
            query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
            inputs = {'input_ids': query_id, 'attention_mask': query_mask}
            query = model_call(args, model, inputs, False)

            code_list = [(batch['code_ids'][i].unsqueeze(0).to(torch.device(args.device)),
                          batch['code_mask'][i].unsqueeze(0).to(torch.device(args.device))) for i in
                         range(0, args.train_batch_size)]

            positive_code_key = code_list.pop(0)
            inputs = {'input_ids': positive_code_key[0], 'attention_mask': positive_code_key[1]}
            positive_code_key = model_call(args, model, inputs, True)

            negative_keys = []
            for code, mask in code_list:
                inputs = {'input_ids': code, 'attention_mask': mask}

                negative_code_key = model_call(args, model, inputs, True)
                negative_keys.append(negative_code_key.clone().detach())

            negative_keys_reshaped = torch.cat(negative_keys[:min(args.num_of_negative_samples, len(negative_keys))],
                                               dim=0)

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
    batch_size = 1
    num_distractors = args.num_of_distractors

    device = args.device
    eval_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

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
        positive_code = np.expand_dims(positive_code, axis=0)
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

    return all_distances

def evaluation(args, model, valid_set):
    logging.info("Evaluate ...")
    print("Evaluate ...")

    batch_size = 1
    eval_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    all_distances = []
    progress_bar = tqdm(eval_dataloader, desc="Evaluation", position=3, leave=True, dynamic_ncols=True)
    for idx, batch in enumerate(progress_bar):

        query_id = batch['doc_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        query_mask = batch['doc_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': query_id, 'attention_mask': query_mask}
        query = model_call(args, model, inputs, False)

        positive_code_key_id = batch['code_ids'][0].to(torch.device(args.device)).unsqueeze(0)
        positive_code_key_mask = batch['code_mask'][0].to(torch.device(args.device)).unsqueeze(0)
        inputs = {'input_ids': positive_code_key_id, 'attention_mask': positive_code_key_mask}
        positive_code_key = model_call(args, model, inputs, True)

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
            negative_code_key = model_call(args, model, inputs, True)
            negative_keys.append(negative_code_key.clone().detach())

        negative_keys_reshaped = torch.cat(negative_keys, dim=0)

        # calc Cosine distance for positive key at first position
        distances = [calculate_cosine_distance(query, positive_code_key)]

        for i in range(len(negative_keys_reshaped)):
            # calc Euclidean distance for negative keys
            distance = calculate_cosine_distance(query, negative_keys_reshaped[i])
            distances.append(distance)

        logging.debug("distances: %s", distances)
        all_distances.append(distances)

    logging.info("***** finished evaluation *****")
    return all_distances


def calculate_cosine_distance(positive_code_key, query):
    # Remove the explicit dimension of size 1 and shift to cpu
    positive_code_key = np.squeeze(positive_code_key.detach().cpu().numpy())
    query = np.squeeze(query.detach().cpu().numpy())

    cosine_similarity = (np.dot(query, positive_code_key) /
                         (np.linalg.norm(query) * np.linalg.norm(positive_code_key)))
    # Compute cosine distance
    return 1 - cosine_similarity


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

    args = parser.parse_args()
    args.dataset = 'codebert-base'
    args.model_name = 'microsoft/codebert-base'
    args.MAX_LEN = 256

    # set minibatchsize to 2 for triplet and contrastive loss
    # and macrobatchsize to (num_accumulation_steps * trainbatchsize) / 2
    if args.loss_function == "triplet" or args.loss_function == "ContrastiveLoss":
        args.num_of_accumulation_steps = (args.num_of_accumulation_steps * args.train_batch_size) / 2
        args.train_batch_size = 2
        args.eval_batch_size = 2

    # Setup CUDA, GPU
    if args.GPU:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    visualisation_dataset = test_dataset.sample(99)
    visualisation_dataset = visualisation_dataset.reset_index(drop=True)

    logging.debug("TRAIN Dataset: %s", train_dataset.shape)
    logging.debug("VAL Dataset: %s", valid_dataset.shape)
    logging.debug("TEST Dataset: %s", test_dataset.shape)

    training_set = CustomDataset(train_dataset, args)
    test_set = CustomDataset(test_dataset, args)
    valid_set = CustomDataset(valid_dataset, args)
    visualization_set = CustomDataset(visualisation_dataset, args)

    # model

    if args.architecture == 'Uni':
        model = UniEncoderModel(args.model_name)
    elif args.architecture == 'Bi':
        model = BiEncoderModel(args.model_name)
    elif args.architecture == 'MoCo':
        model = MoCoModel(args.model_name)
    else:
        exit("Learning architecture not supported")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    model.to(torch.device(args.device))

    # visualize
    # visualize(args, model, visualization_set, True)

    # train
    # train_losses, val_losses = train(args, model, optimizer, training_set, valid_set)

    # visualize train and val losses
    # visualize_losses(train_losses, val_losses, args)

    # visualize again
    # visualize(args, model, visualization_set, False)

    # evaluate
    distances = predict_distances(args, model, test_set)

    mrr = calculate_mrr_from_distances(distances)
    logging.info(f"MRR: {mrr}")



    # Calculate runtime duration in seconds
    end_time = time.time()
    runtime_seconds = end_time - start_time

    # Convert runtime duration to hours, minutes, and seconds
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print or log the runtime
    print(f"Program runtime: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    runtime = f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    # write mrr to file
    write_mrr_to_file(args, mrr, runtime)


if __name__ == '__main__':
    main()
