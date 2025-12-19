import torch.nn as nn

def get_loss(label_smoothing):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)