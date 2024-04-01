import torch


def prox_l1(v, lam):
    result = torch.clamp(v - lam, min=0) - torch.clamp(-v - lam, min=0)
    return result


def prox_l12(v, lam):
    # 0.5 * |x|_1 + 0.5 * (1/2 |x|_2^2)
    result = torch.clamp(v - lam / 2, min=0) - torch.clamp(-v - lam / 2, min=0)
    result = 1 / (1 + lam / 2) * result
    return result
