import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer


class CustomDataset(TensorDataset):

    def __init__(self, dataframe, args):
        self.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        self.text = dataframe.text
        # self.targets = dataframe.labels
        self.max_len = args.MAX_LEN

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
            padding='max_length',
            return_token_type_ids=False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }