import argparse
import time

from torch.utils.data import DataLoader

from losses import triplet_loss, contrastive_loss, info_nce_loss
from models import UniEncoderModel, BiEncoderModel, MoCoModel
from utils import *

LOSS_FUNCTIONS = ['triplet', 'InfoNCE', 'ContrastiveLoss']
ARCHITECTURES = ['Uni', 'Bi', 'MoCo']


def train(args, model, optimizer, training_set, valid_set):
    """
    The training loop

    :param args:
    :param model:
    :param optimizer:
    :param training_set:
    :param valid_set:
    :return:
    """
    logging.info("Start training ...")
    all_train_mean_losses = []
    all_val_mean_losses = []
    for epoch in tqdm(range(1, args.num_train_epochs + 1), desc="Epochs", dynamic_ncols=True, position=0, leave=True):
        logging.info("training epoch %d", epoch)

        # random.shuffle(training_set)
        train_dataloader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)

        all_losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Train", position=1, leave=True, dynamic_ncols=True)
        for idx, batch in enumerate(progress_bar):

            logging.debug("____________________________________________________________")
            logging.debug(f"idx: {idx}")
            if batch['code_ids'].size(0) < args.batch_size:
                logging.debug("continue")
                continue

            negative_keys_reshaped, positive_code_key, query = compute_embeddings_train(args, batch, model)

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

        validation_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
        all_val_losses = []
        if args.do_validation:

            progress_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch} eval ", position=2, leave=True,
                                dynamic_ncols=True)
            for idx, batch in enumerate(progress_bar):
                if batch['code_ids'].size(0) < args.batch_size:
                    logging.debug("continue")
                    continue

                negative_keys_reshaped, positive_code_key, query = compute_embeddings_train(args, batch, model,
                                                                                            validation=True)

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
    """
    Evaluating the model after training on the evaluation set

    2 Versions depending on the architecture. One for MoCo and the other one for Uni- and Bi-encoder setups

    :param args:
    :param model:
    :param test_set:
    :return:
    """
    logging.info("Start Evaluation...")
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
        if args.architecture == 'MoCo':
            code_emb, _ = model_call(args, model, inputs, True)
        else:
            code_emb = model_call(args, model, inputs, True)
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

    parser.add_argument("--lang", default='python', type=str, required=False, help="Language of the code")

    parser.add_argument("--batch_size", default=16, type=int, required=False, help="Training batch size")

    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False, help="Learning rate")

    parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="Number of training epochs")

    parser.add_argument("--momentum", default=0.999, type=float, required=False, help="Momentum parameter")

    parser.add_argument("--train_size", default=0.8, type=float, required=False, help="percentage of train dataset used"
                                                                                      "for training")

    parser.add_argument("--data_path", default='./data/', type=str, required=False, help="Path to data folder")

    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    parser.add_argument("--num_of_accumulation_steps", default=16, type=int, required=False,
                        help="Number of accumulation steps")

    parser.add_argument("--num_of_negative_samples", default=15, type=int, required=False, help="Number of negative "
                                                                                                "samples")
    parser.add_argument("--num_of_distractors", default=99, type=int, required=False, help="Number of distractors")
    parser.add_argument("--queue_length", default=4096, type=int, required=False, help="MoCo queue length")
    parser.add_argument("--GPU", required=False, help="specify the GPU which should be used")
    parser.add_argument("--do_generalisation", default=True, type=bool, required=False)
    parser.add_argument("--do_validation", default=True, type=bool, required=False)
    parser.add_argument("--do_visualization", default=True, type=bool, required=False)

    args = parser.parse_args()
    args.dataset = 'codebert-base'
    args.model_name = 'microsoft/codebert-base'
    args.MAX_LEN = 256

    # set mini-batchsize to 2 for triplet and contrastive loss
    # and macro-batchsize to (num_accumulation_steps * train-batchsize) / 2
    # and set num of negatives for triplet and ContrastiveLoss
    if args.loss_function == "triplet" or args.loss_function == "ContrastiveLoss":
        args.batch_size = 2
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

    #####################
    # visualize
    #####################
    if args.do_visualization:
        print("visualize embeddings__________")
        visualize(args, model, visualization_set, True)

    #####################
    # evaluate
    #####################
    print("evaluate model_________________")
    distances = predict_distances(args, model, test_set)

    mrr_before_train = calculate_mrr_from_distances(distances)
    logging.info(f"MRR Before training: {mrr_before_train}")

    #####################
    # test generalisation
    #####################
    generalisation_mrr = 0
    if args.do_generalisation:
        logging.info("Start Generalisation...")
        generalisation_df = load_stat_code_search_dataset(args)

        generalisation_set = CustomDataset(generalisation_df, args)
        distances = predict_distances(args, model, generalisation_set)

        generalisation_mrr_before_train = calculate_mrr_from_distances(distances)
        logging.info(f"Generalisation MRR: {generalisation_mrr}")
        print(f"gen before train: {generalisation_mrr_before_train}")

    #####################
    # train
    #####################
    print("train model____________________")
    train_losses, val_losses = train(args, model, optimizer, training_set, valid_set)

    #####################
    # visualize
    #####################
    if args.do_visualization:
        # visualize train and val losses
        print("visualize losses_______________")
        visualize_losses(train_losses, val_losses, args)

        # visualize embeddings
        print("visualize embeddings___________")
        visualize(args, model, visualization_set, False)

    #####################
    # evaluate
    #####################
    print("evaluate model_________________")
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
    write_mrr_to_file(args, mrr_before_train, mrr, generalisation_mrr, runtime)


if __name__ == '__main__':
    main()
