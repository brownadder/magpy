import numpy as np

from .methods import eye, is_square, kron,sigmax,sigmay,sigmaz

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

    def __init__(self, *args, **kwargs):
        """
        Construct matrix representing the quantum operator.
        """

        if not args and not kwargs:
            raise TypeError("input cannot be empty")

        elif len(args) == 1 and isinstance(args[0],int) and kwargs:
            matricesholder = args[0] * [np.zeros(2,dtype=complex)+eye(2)]
            Hmatrix = np.zeros(2**args[0],dtype=complex)

            if 'precession_X' in kwargs:
                for i in range(args[0]):
                    matrices = np.copy(matricesholder)
                    matrices[i] = sigmax()
                    Hmatrix = Hmatrix + kwargs['precession_X'][i]*kron(matrices)

            if 'precession_Y' in kwargs:
                for i in range(args[0]):
                    matrices = np.copy(matricesholder)
                    matrices[i] = sigmay()
                    Hmatrix = Hmatrix + kwargs['precession_Y'][i]*kron(matrices)

            if 'precession_Z' in kwargs:
                for i in range(args[0]):
                    matrices = np.copy(matricesholder)
                    matrices[i] = sigmaz()
                    Hmatrix = Hmatrix + kwargs['precession_Z'][i]*kron(matrices)

            if 'coupling_matrix_XX' in kwargs:
                if (
                    (kwargs['coupling_matrix_XX'].transpose() == kwargs['coupling_matrix_XX']).all() and
                    not np.any(kwargs['coupling_matrix_XX'].diagonal())
                ):
                    for j in range(args[0]-1):
                        k = j+1
                        for k in range(j+1,args[0]):
                            matrices = np.copy(matricesholder)
                            matrices[j] = sigmax()
                            matrices[k] = sigmax()
                            Hmatrix = Hmatrix + kwargs['coupling_matrix_XX'][j,k]*kron(matrices)
                else:
                    raise ValueError("invalid XX couping matrix")

            if 'coupling_matrix_YY' in kwargs:
                if (
                    (kwargs['coupling_matrix_YY'].transpose() == kwargs['coupling_matrix_YY']).all() and
                    not np.any(kwargs['coupling_matrix_YY'].diagonal())
                ):
                    for j in range(args[0]-1):
                        k = j+1
                        for k in range(j+1,args[0]):
                            matrices = np.copy(matricesholder)
                            matrices[j] = sigmay()
                            matrices[k] = sigmay()
                            Hmatrix = Hmatrix + kwargs['coupling_matrix_YY'][j,k]*kron(matrices)
                else:
                    raise ValueError("invalid YY couping matrix")
            
            if 'coupling_matrix_ZZ' in kwargs:
                if (
                    (kwargs['coupling_matrix_ZZ'].transpose() == kwargs['coupling_matrix_ZZ']).all() and
                    not np.any(kwargs['coupling_matrix_ZZ'].diagonal())
                ):
                    for j in range(args[0]-1):
                        k = j+1
                        for k in range(j+1,args[0]):
                            matrices = np.copy(matricesholder)
                            matrices[j] = sigmaz()
                            matrices[k] = sigmaz()
                            Hmatrix = Hmatrix + kwargs['coupling_matrix_ZZ'][j,k]*kron(matrices)
                else:
                    raise ValueError("invalid ZZ couping matrix")
            
            self.data = Hmatrix

        elif len(args) == 1 and is_square(args[0]) and not kwargs:
            # single ndarray

            self.data = args[0]

        elif len(args) > 1 and isinstance(args[1], tuple) and not kwargs:
            # list of tuples (with pos and ndarray)

            matrices = args[0] * [eye(2)]

            for spin in args[1:]:
                matrices[spin[0] - 1] = spin[1]

            self.data = kron(matrices)
            
        elif len(args) > 1 and args[0] >= args[1] and is_square(args[2]) and not kwargs:
            # multi-spin system with one spin specified

            matrices = args[0] * [eye(2)]
            matrices[args[1] - 1] = args[2]
            self.data = kron(matrices)

        else:
            raise ValueError("invalid input")

    def __call__(self):
        return self.data

    def is_hermitian(self):
        return np.array_equal(self.data.conj().T, self.data)