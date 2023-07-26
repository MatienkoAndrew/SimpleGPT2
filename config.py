import torch

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 64,
    'n_epochs': 2,
    'max_length': 128
}
