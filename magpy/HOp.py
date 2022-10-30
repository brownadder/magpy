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

        >>> mp.HOp(operations = (mp.sigmax()))
        >>> mp.HOp(operations = (1, 1, mp.sigmax()))

    Two spins:
        H = sigmax x Id

        >>> mp.HOp(operations = (2, 1, mp.sigmax()))

    Two spins interacting:
        H = sigmax x sigmay

        >>> mp.HOp(operations = (2, (1,mp.sigmax()), (2,mp.sigmay())))

    """

    def __init__(
        self, 
        operations = None,
        matrix = None,
        ):
        """
        Construct matrix representing the quantum operator.
        """

        if operations == None:
        
            if isinstance(matrix,np.ndarray) and is_square(matrix):
                # when input matrix is a numpy array
                self.data = matrix

            elif isinstance(matrix, list) and is_square(np.array(matrix)):
                # when input matrix is a list, convert it to numpy array
                self.data = np.array(matrix, dtype=complex)

            elif matrix is None:
        
                raise TypeError("input cannot be empty")

        elif len(operations) == 1 and is_square(operations[0]):
            # single ndarray

            if matrix is None:
                self.data = operations[0]

            else:
                raise ValueError("invalid input")

        elif isinstance(operations[1], tuple):
            # list of tuples (with pos and ndarray)

            if matrix is None:

                matrices = operations[0] * [eye(2)]

                for spin in operations[1:]:
                    matrices[spin[0] - 1] = spin[1]

                self.data = kron(matrices)
                
            else:
                raise ValueError("invalid input")
            
        elif operations[0] >= operations[1] and is_square(operations[2]):
            # multi-spin system with one spin specified

            if matrix is None:

                matrices = operations[0] * [eye(2)]
                matrices[operations[1] - 1] = operations[2]
                self.data = kron(matrices)

            else:
                raise ValueError("invalid input")

        else:
            raise ValueError("invalid input")

    def __call__(self):
        return self.data

    def is_hermitian(self):
        return np.array_equal(self.data.conj().T, self.data)