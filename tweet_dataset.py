import torch
import numpy as np
from config import CONFIG


class TweetDataset(torch.utils.data.Dataset):
    """
    Класс TweetDataset предназначен для создания датасета из твитов.
    Этот класс наследует от torch.utils.data.Dataset и переопределяет методы __init__, __getitem__ и __len__.
    """
    def __init__(self, part, dataset, tokenizer):
        self.part = part
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = CONFIG['max_length']
        
        self.labels = np.unique(dataset[part]["label"])
        self.label2num = {l: num for num, l in enumerate(self.labels)}
        
    def __getitem__(self, idx):
        """
        Return dict with tokens, attention_mask and label
        """
        text = self.dataset[self.part]['text'][idx]
        label = self.dataset[self.part]['label'][idx]
        
        tokenizer_output = self.tokenizer.encode_plus(
            text, 
            max_length=self.max_length, # максимальная длина текста
            padding="max_length", # надо ли добавлять паддинг в конце?
            return_tensors="pt", # возвращает pytorch тензоры
        )

        target = torch.tensor(self.label2num[label])
        return {
            "input_ids": tokenizer_output['input_ids'], 
            "mask": tokenizer_output['attention_mask'],
            "target": target
        }
        
    def __len__(self):
        """
        Return length of dataset
        """
        return len(self.dataset[self.part])
