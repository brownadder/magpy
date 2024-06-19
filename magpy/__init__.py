from .core import PauliString, X, Y, Z, Id, FunctionProduct, HamiltonianOperator, kron, frobenius, timegrid
from .solver import System

__all__ = [
    'PauliString', 'X', 'Y', 'Z', 'Id', 'FunctionProduct', 'HamiltonianOperator', 'kron', 'frobenius', 'timegrid',
    'System'
]


def set_device(device):
    """Set the device to use when evaluating MagPy objects and
    performing calculations. The default is `cpu`.

    Parameters
    ----------
    device : str
        Device name
    """

    from ._device import _DEVICE_CONTEXT
    _DEVICE_CONTEXT.device = device
