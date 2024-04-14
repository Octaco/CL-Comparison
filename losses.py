import torch
from info_nce import InfoNCE


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
