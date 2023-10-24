import torch

def saliency_mask(saliency: torch.Tensor, r=.5, k=None):
    if k is None:
        k = int(r * saliency.numel())
    saliency_abs = saliency.abs()
    return (saliency_abs >= saliency_abs.flatten().topk(k).values.min())