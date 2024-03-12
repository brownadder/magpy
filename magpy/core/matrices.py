import torch
import functools


def kron(*args):  # TODO: Replace this with expsolve.kron
    return functools.reduce(torch.kron, args)


def frobenius(a, b):  # TODO: Parallelise this. Replace 'out' with tensor
    try:
        out = []
        for x in a:
            out.append(torch.trace(torch.conj(torch.transpose(x, 0, 1)) @ b))
        return out

    except AttributeError:
        return torch.trace(torch.conj(torch.transpose(a, 0, 1)) @ b)


def commutator(A, B):
    return A*B - B*A
