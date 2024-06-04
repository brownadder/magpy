import functools
import torch
from .._device import _DEVICE_CONTEXT

def kron(*args):
    return functools.reduce(torch.kron, args).to(_DEVICE_CONTEXT.device)


def frobenius(a, b):
    try:
        return torch.vmap(torch.trace)(torch.matmul(torch.conj(torch.transpose(a, 1, -1)), b)) \
            .to(_DEVICE_CONTEXT.device)
    except RuntimeError:
        return torch.trace(torch.conj(torch.transpose(a, 0, 1)) @ b).to(_DEVICE_CONTEXT.device)


def timegrid(start, stop, step):
    return torch.arange(start, stop + step, step).to(_DEVICE_CONTEXT.device)
