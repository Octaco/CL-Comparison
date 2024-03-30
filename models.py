import random

import torch
from transformers import RobertaModel


class UniEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(UniEncoderModel, self).__init__()
        self.model = RobertaModel.from_pretrained(model_name)

    def get_hidden_size(self):
        return self.model.config.hidden_size

    def get_parameter(self, **kwargs):
        return self.model.parameters()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[1]  # using pooled output


class BiEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(BiEncoderModel, self).__init__()
        self.model_q = UniEncoderModel(model_name)  # for the NL query
        self.model_k = UniEncoderModel(model_name)  # for the Code keys
        output_dim = self.model_k.get_hidden_size()
        self.prediction_head = torch.nn.Linear(output_dim, output_dim)  # Linear layer for prediction head

    def forward(self, is_code_key, input_ids, attention_mask):
        if is_code_key:
            output = self.model_k(input_ids=input_ids, attention_mask=attention_mask)
            # logits = self.prediction_head(output)
            return output
        else:  # no gradient for the queries
            with torch.no_grad():
                output = self.model_q(input_ids=input_ids, attention_mask=attention_mask)
                return output


class MoCoModel(torch.nn.Module):
    """
    dim = max model dim
    k = queue length
    m = momentum
    """

    def __init__(self, args, k=4096):
        super(MoCoModel, self).__init__()
        self.k = k
        self.m = args.momentum
        self.num_of_negative_samples = args.num_of_negative_samples
        self.num_of_distractors = args.num_of_distractors
        self.model_q = RobertaModel.from_pretrained(args.model_name)
        self.model_k = RobertaModel.from_pretrained(args.model_name)


        # queue
        self.queue = []

    def _dequeue_and_enqueue(self, keys):

        if len(self.queue) < self.k:
            self.queue.append(keys)
        else:
            self.queue.pop(0)
            self.queue.append(keys)

    def _momentum_update(self):
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
            param_k.requires_grad = False

    def forward(self, is_code_key, input_ids, attention_mask, visualisation=False):

        if is_code_key:
            with torch.no_grad():
                self._momentum_update()

                positive_key = self.model_k(input_ids=input_ids, attention_mask=attention_mask)[1]  # using pooled output

                if visualisation:
                    negative_keys = random.sample(self.queue, min(len(self.queue), self.num_of_distractors))
                else:
                    negative_keys = random.sample(self.queue, min(len(self.queue), self.num_of_negative_samples))

                self._dequeue_and_enqueue(positive_key)

                return positive_key, negative_keys

        else:
            query = self.model_q(input_ids=input_ids, attention_mask=attention_mask)[1]  # using pooled output
            return query
