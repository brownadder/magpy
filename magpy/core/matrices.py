import torch
import functools


def kron(*args):  # TODO: Replace this with expsolve.kron
    return functools.reduce(torch.kron, args)
