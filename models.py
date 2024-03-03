import torch
from transformers import RobertaModel, RobertaTokenizer


class UniEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(UniEncoderModel, self).__init__()
        self.model = RobertaModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class BiEncoderModel(torch.nn.Module):
    def __init__(self, model_name):
        super(BiEncoderModel, self).__init__()
        self.model = RobertaModel.from_pretrained(model_name)
        output_dim = self.model.config.hidden_size
        self.prediction_head = torch.nn.Linear(output_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.prediction_head(pooled_output)
        return logits


class MoCoModel(torch.nn.Module):
    def __init__(self, model_name, dim=768, k=4096, m=0.999):
        super(MoCoModel, self).__init__()
        self.k = k
        self.m = m
        self.model_q = RobertaModel.from_pretrained(model_name)
        self.model_k = RobertaModel.from_pretrained(model_name)
        self._momentum_update()
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.model_q.config.hidden_size, dim))

        # queue
        self.register_buffer("queue", torch.randn(dim, k))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)

        # queue pointer
        self.pointer = 0

    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating
        keys = keys.detach()
        batch_size = keys.shape[0]

        # replace the keys at the pointer
        self.queue[:, self.pointer:self.pointer + batch_size] = keys.T
        self.pointer = (self.pointer + batch_size) % self.k

    def _momentum_update(self):
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            param_k.requires_grad = False

    def forward(self, input_ids, attention_mask):
        q, _ = self.model_q(input_ids=input_ids, attention_mask=attention_mask)
        q = self.projection_head(q)
        q = torch.nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update()
            k, _ = self.model_k(input_ids=input_ids, attention_mask=attention_mask)
            k = self.projection_head(k)
            k = torch.nn.functional.normalize(k, dim=1)

        self._dequeue_and_enqueue(k)

        # positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits
        logits = torch.cat([l_pos, l_neg], dim=1)

        return logits
