import torch
import functools

def kron(*args):  # TODO: Replace this with expsolve.kron
    return functools.reduce(torch.kron, args)


def frobenius(a, b):
    try:
        return torch.vmap(torch.trace)(torch.matmul(torch.conj(torch.transpose(a, 1, -1)), b))
    except RuntimeError:
        return torch.trace(torch.conj(torch.transpose(a, 0, 1)) @ b)


def commutator(A, B):
    return A*B - B*A


def timegrid(start, stop, step):
    return torch.arange(start, stop + step, step)
