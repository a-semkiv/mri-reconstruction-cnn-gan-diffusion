import torch
import torch.nn.functional as F


def gan_loss(prediction, is_real):
    if is_real:
        target = torch.ones_like(prediction)
    else:
        target = torch.zeros_like(prediction)
    return F.binary_cross_entropy_with_logits(prediction, target)


def l1_loss(prediction, target):
    return F.l1_loss(prediction, target)
