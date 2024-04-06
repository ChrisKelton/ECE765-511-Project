import torch


def imshow(inp: torch.Tensor, title=None):
    """Display image for Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    