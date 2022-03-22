import torch

def collate_fn_pad_batch(data):
    """Pad data in a batch.
    Parameters
    ----------
    data : list((tensor, int), )
        data and label in a batch
    Returns
    -------
    tuple(tensor, tensor)
    """
    num_items = len(data[0])
    max_len = max([i[0].shape[0] for i in data])
    labels = torch.tensor([i[1] for i in data], dtype=torch.long)
    padded = torch.zeros((len(data), max_len), dtype=torch.long)
    if num_items == 3:
        weights = torch.tensor([i[2] for i in data], dtype=torch.float)
    # randomizing might be better
    for i, _ in enumerate(padded):
        padded[i][:data[i][0].shape[0]] = data[i][0]
    if num_items == 3:    
        return padded, labels, weights
    else:
        return padded, labels


def max_len_pad(data):
    """Pad data globally.
    Parameters
    ----------
    data : list((tensor, int), )
        data and label in a batch
    Returns
    -------
    tuple(tensor, tensor)
    """
    max_len = -1
    for sample in data:
        print(sample[0])
    num_items = len(data[0])
    labels = torch.tensor([i[1] for i in data], dtype=torch.long)
    padded = torch.zeros((len(data), max_len), dtype=torch.long)
    if num_items == 3:
        weights = torch.tensor([i[2] for i in data], dtype=torch.float)
    # randomizing might be better
    for i, _ in enumerate(padded):
        padded[i][:data[i][0].shape[0]] = data[i][0]
    if num_items == 3:    
        return padded, labels, weights
    else:
        return padded, labels