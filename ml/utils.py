import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn_mlp(batch):
    x_a_list, x_b_list, y_list = zip(*batch)
    return torch.stack(x_a_list), torch.stack(x_b_list), torch.stack(y_list)

def collate_fn(batch):
    x_a_list, x_b_list, y_list = zip(*batch)
    return (
        pad_sequence(x_a_list, batch_first=True, padding_value=0),
        pad_sequence(x_b_list, batch_first=True, padding_value=0),
        torch.stack(y_list)
    )
