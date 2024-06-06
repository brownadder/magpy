import functools
import torch
from .._device import _DEVICE_CONTEXT

def kron(*args):
    """Compute the Kronecker product of the input arguments.

    Returns
    -------
    Tensor
        Resultant product
    """

    return functools.reduce(torch.kron, args).to(_DEVICE_CONTEXT.device)


def frobenius(a, b):
    """Compute the Frobenius inner product of `a` and `b`. 

    If `a` is a 3D tensor and `b` is a 2D tensor, then the inner product is
    batched across `a`. Otherwise `a` and `b` must both be 2D tensors.

    Parameters
    ----------
    a : Tensor
        First argument(s)
    b : Tensor
        Second argument

    Returns
    -------
    Tensor
        Resultant (batch) product
    """

    try:
        return torch.vmap(torch.trace)(torch.matmul(torch.conj(torch.transpose(a, 1, -1)), b)) \
            .to(_DEVICE_CONTEXT.device)
    except RuntimeError:
        return torch.trace(torch.conj(torch.transpose(a, 0, 1)) @ b).to(_DEVICE_CONTEXT.device)


def timegrid(start, stop, step):
    """Create a grid of values across the specified interval with the
    specified spacing.

    Parameters
    ----------
    start : float
        Start of interval
    stop : float
        End of interval
    step : float
        Spacing of points in interval

    Returns
    -------
    Tensor
        Grid of values
    """

    return torch.arange(start, stop + step, step).to(_DEVICE_CONTEXT.device)
