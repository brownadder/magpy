import numpy as np

from .methods import eye, is_square, kron

class HOp:
    """
    Represents a constant Hamiltonian operator. This can be for a system 
    containing one spin, for a single spin in part of a larger system, or for 
    an interactive component of a larger system.

    Attributes
    ----------
    data : ndarray
        Matrix representing the quantum operator.

    Examples
    --------
    One spin:
        H = sigmax

        >>> mp.HOp(mp.sigmax())
        >>> mp.HOp(1, 1, mp.sigmax())

    Two spins:
        H = sigmax x Id

        >>> mp.HOp(2, 1, mp.sigmax())

    Two spins interacting:
        H = sigmax x sigmay

        >>> mp.HOp(2, (1,mp.sigmax()), (2,mp.sigmay()))

    """

    def __init__(self, *args):
        """
        Construct matrix representing the quantum operator.
        """

        if not args:
            raise TypeError("args cannot be empty")

        if len(args) == 1 and isinstance(args[0], np.ndarray):
            # single array
            if not is_square(args[0]):
                raise ValueError("array must be square")
            self.data = args[0]

        elif list(map(type, args)) == [int, int, np.ndarray]:
            # multi-spin system with one spin specified
            if not is_square(args[2]):
                raise ValueError("array must be square")
            if args[0] <= 0 or args[1] <= 0 or args[0] < args[1]:
                raise ValueError("index must be <= spin number and both must be positive")
            
            matrices = args[0] * [eye(2)]
            matrices[args[1] - 1] = args[2]
            self.data = kron(matrices)

        elif list(map(type, args)) == [int] + [tuple for _ in args[1:]]:
            # spin number + list of tuples (index and array)
            if not args[1:]:
                raise ValueError("at least one tuple of index and array must be specified")
            if args[0] <= 0:
                raise ValueError("spin number must be positive")

            for arg in args[1:]:
                if not isinstance(arg[1], np.ndarray):
                    raise TypeError("matrix must be ndarray")
                if arg[0] > args[0] or arg[0] <= 0:
                    raise ValueError("all indices must be <= spin number and all must be positive")

            matrices = args[0] * [eye(2)]
            for spin in args[1:]:
                matrices[spin[0] - 1] = spin[1]
            self.data = kron(matrices)
            
        else:
            raise ValueError("invalid args specified")

    def __call__(self):
        return self.data

    def is_hermitian(self):
        return np.array_equal(self.data.conj().T, self.data)