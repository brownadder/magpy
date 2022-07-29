import numpy as np
import scipy.integrate


def sigmax():
    return np.array([[0, 1], [1, 0]])


def sigmay():
    return np.array([[0, -1j], [1j, 0]])


def sigmaz():
    return np.array([[1, 0], [0, -1]])


def vec(mat):
    """
    Return vectorised form of input using column-major (Fortran) ordering.

    Parameters
    ----------
    mat : ndarray
        Matrix.

    Returns
    -------
    ndarray
        Vector.

    """

    return np.asarray(mat).flatten('F')


def unvec(vec, c=None):
    """
    Return unvectorised vector using column-major (Fortran) ordering.

    Parameters
    ----------
    vec : ndarray
        Vector of elements.
    c : int, optional
        Desired length of columns in matrix. Infers square matrix if so.
        The default is None.

    Returns
    -------
    ndarray
        Matrix.

    """
    vec = np.array(vec)

    if (len(vec) % 2 != 0):
        # odd number of elements
        if (len(vec) == 1):
            return vec
        else:
            print("Error: odd number of elements in vector. \
                  Cannot form matrix.")
            return None
    elif c is None:
        if (np.sqrt(len(vec)).is_integer()):
            # matrix is square
            c = int(np.sqrt(len(vec)))
        else:
            print("Error: vector cannot form a square matrix. \
                  Please provide a column length, c.")
            return None
    elif (not (len(vec) / c).is_integer()):
        # c does not divide length of vec
        print("Error: value of c is invalid. \
              Cannot split vector evenly into columns of length c")
        return None

    # number of rows
    n = int(len(vec) / c)

    return vec.reshape((c, n), order='F')


def liouvillian(H):
    """
    Return Liouvillian of a Hamiltonian.

    Parameters
    ----------
    H : ndarray
        Square matrix with dimension n.

    Returns
    -------
    ndarray
        Square matrix with dimension n^2.

    """
    n = H.shape[0]

    return -1j * (np.kron(np.eye(n), H) - np.kron(H.T, np.eye(n)))


def commutator(A, B, kind="normal"):
    """
    Return commutator of kind of A and B.

    Parameters
    ----------
    A : ndarray
        Square array.
    B : ndarray
        Square array.
    kind : str, optional
        kind of commutator (normal, anti), The default is "normal".

    Returns
    -------
    ndarray
        Commutator of A and B.

    """
    if kind == "normal":
        return A@B - B@A
    elif kind == "anti":
        return A@B + B@A
    else:
        raise TypeError("Unknown commutator kind " + str(kind))


def kron(*args):
    """
    Return Kronecker product of input arguments.

    Returns
    -------
    ndarray
        Kronecker product.

    Raises
    ------
    TypeError
        No input arguments.

    """
    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], list):
        # input of the form [a,b,...]
        mlist = args[0]
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        # single
        if len(args[0].shape) == 2:
            return args[0]
        else:
            # ndarray
            mlist = args[0]
    else:
        mlist = args

    out = mlist[0]
    for m in mlist[1:]:
        out = np.kron(out, m)

    return out


def linspace(start, stop, step, dtype=None):
    """
    Return numbers spaced by specified step over specified interval.

    Parameters
    ----------
    start : array_like
        Starting value of sequence.
    stop : array_like
        End value of sequence.
    step : array_like
        Amount by which to space points in sequence.
    dtype : dtype, optional
        The type of the output array. If dtype is not given,then the data type 
        is inferred from arguments. The default is None.

    Returns
    -------
    ndarray
        Equally spaced numbers as specified.

    """

    return (np.linspace(start, stop, int((stop - start) / step) + 1)
            .astype(dtype))


def frobenius(a, b):
    """
    Return Frobenius/trace inner product of a and b.
    Applied element-wise if a is not single.

    Parameters
    ----------
    a : ndarray
        Square array or list/array of square arrays.
    b : ndarray
        Square array.

    Returns
    -------
    ndarray or scalar
        The value(s) of the trace of a times b.

    Examples
    --------
    >>> mp.Frobenius(np.eye(2), np.ones((2,2)))
    (2+0j)
    >>> a = [np.eye(2), np.ones((2,2))]
    >>> mp.Frobenius(a, np.ones((2,2)))
    array([2.+0.j, 4.+0.j])

    """
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=complex)

    try:
        # a is an array
        t = []
        for x in a:
            t.append(np.trace(x.conj().T @ b))
        return np.asarray(t)

    except:
        # a is single
        return np.trace(a.conj().T @ b)


class HOp: #TODO
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

        >>> mp.HOp(1, 1, mp.sigmax())

    Two spins:
        H = sigmax x Id

        >>> mp.HOp(2, 1, mp.sigmax())

    Two spins interacting:
        H = sigmax x sigmay

        >>> mp.HOp(2, (1,mp.sigmax()), (2,mp.sigmay()))

    """

    def __init__(self, n, *args):
        """
        Construct matrix representing the quantum operator.

        Parameters
        ----------
        n : int
            Number of spins in system. Must be greater than or equal to 1.
        """
        if n > 1:
            # multiple spins
            matrices = n * [np.eye(2)]
            
            if type(args[0]) == type(1):
                matrices[args[0] - 1] = args[1]
            else:
                for spin in args:
                    matrices[spin[0] - 1] = spin[1]

            self.data = kron(matrices)
        else:
            # single spin
            if type(args[0]) == type((1,)):
                self.data = args[0][1]
            else:
                self.data = args[1]

    def __call__(self):
        return self.data

    def is_hermitian(self):
        return np.array_equal(self.data.conj().T, self.data)


class System:
    """
    Represents a quantum system with a specified Hamiltonian, allowing for the
    simulation of the system using a two-term Magnus expansion without 
    requiring re-calculation of the Liouvillians each time.

    Parameters
    ----------
    H : dict
        The Hamiltonian of the system, which takes the form:

            H = { f_1 : H_1, f_2 : H_2, ... },
        
        where each f_i is a function and each H_i is a HOp object. 

    Attributes
    ----------
    H1 : dict
        Pre-calculated Liouvillians and corresponding functional coefficients
        for the first term.

        Takes the form: 
            
            { f_1 : L(H_1), f_2 : L(H_2), ... },

        where f_i are functions (or constants) and L(H_i) are liouvillians of 
        the corresponding matrices. L(H_i) are square ndarrays.

    H2 : dict
        Pre-calculated Liouvillians and corresponding functional coefficients
        for the second term.

        Takes the form: 
        
            { f_1(y)*f_2(x) : L([H_1, H_2]), 
              f_1(y)*f_3(x) : L([H_1, H_3]), ... },
                
        where each entry is the two-variable function and its corresponding
        liouvillian of a commutator. L([H_i, H_j]) are square ndarrays.

    Examples
    --------
    Single spin system: 
        H(t) = f(t)*sigmax + g(t)*sigmay

        >>> H = {f : mp.HOp(1,1,mp.sigmax()), g : mp.HOp(1,1,mp.sigmay())}

    Two spin system: 
        H(t) = f(t)*(sigmax x Id) + g(t)*(Id x sigmay)

        >>> H = {f : mp.HOp(2,1,mp.sigmax()), g : mp.HOp(2,2,mp.sigmay())}

    Two spin system with repeated coefficient:
        H(t) = f(t)*(sigmax x Id) + f(t)*(Id x sigmay) + g(t)*(sigmaz x Id)

        >>> H = {f : [mp.HOp(2,1,mp.sigmax()), mp.HOp(2,2,mp.sigmay())], 
        g : mp.HOp(2,1,mp.sigmaz())}

    Interacting systems:
        H(t) = f(t)*(sigmax x Id) + g(t)*(sigmax x sigmay)

        >>> H = {f : mp.HOp(2,1,mp.sigmax()), 
        g : mp.HOp(2,(1,mp.sigmax()),(2,mp.sigmay()))}

        H(t) = f(t)*(Id x sigmax x Id) + g(t)*(sigmax x sigma x Id)

        >>> H = {f : mp.HOp(3,2,mp.sigmax()), 
                 g : mp.HOp(3,(1,mp.sigmax()),(2,mp.sigmax()))}

    """

    def __init__(self, H):
        """
        Perform pre-calculations for a two term Magnus expansion, and stored 
        in H1 and H2 to reduce reptitive calculations.

        Parameters
        ----------
        H : dict
            Hamiltonian of the system, which takes the form:

                H = { f_1 : H_1, f_2 : H_2, ... },
            
            where f_i are functions and H_i are HOp objects.

        """
        self.H1 = self.__setup_first_term(H)
        self.H2 = self.__setup_second_term(H)

    def update_hamiltonian(self, H):
        self.__init__(H)

    def __setup_first_term(self, H):
        """
        Pre-calculate Liouvillians for matrices (HOp) in H 
        for first term.
        """
        H1 = {}

        for coeff in H:
            if isinstance(H[coeff], HOp):
                # convert single HOp to list of (single) HOp
                H[coeff] = [H[coeff]]

            matrices = []
            for matrix in H[coeff]:
                matrices.append(liouvillian(matrix()))
            
            H1[coeff] = matrices
        
        return H1

    def __setup_second_term(self, H):
        """
        Pre-calculate Liouvillians for matrices (HOp) in H 
        for first term.
        """
        temp = []

        for coeff in H:
            if isinstance(H[coeff], list):
                for matrix in H[coeff]:
                    temp.append((coeff, matrix()))
            else:
                temp.append((coeff, H[coeff]()))

        n = len(temp)
        H2 = {}

        for i in range(n):
            for j in range(n):
                if i != j:
                    if (isinstance(temp[i][0], (int, float)) 
                        and isinstance(temp[j][0], (int, float))):
                        f = temp[i][0] * temp[j][0]
                    elif isinstance(temp[i][0], (int, float)):
                        def f(y, x): return temp[i][0] * temp[j][0](x)
                    elif isinstance(temp[j][0], (int, float)):
                        def f(y, x): return temp[i][0](y) * temp[j][0]
                    else:
                        def f(y,x): return temp[i][0](y) * temp[j][0](x)

                    H2[f] = 1j * liouvillian(commutator(temp[i][1](), 
                                                                temp[j][1]()))
        
        return H2

    def __eval_first_term(self, t0, tf):
        """
        Evaluate first term of Magnus expansion over specified time
        interval using pre-calculated terms.
        """
        total = 0

        for coeff in self.H1:
            if isinstance(coeff, (int, float)):
                for matrix in self.H1[coeff]:
                    total = total + matrix*coeff*(tf - t0)
            else:
                c = scipy.integrate.quad(coeff, t0, tf)[0]
                for matrix in self.H1[coeff]:
                    total = total + c*matrix
        
        return total

    def __eval_second_term(self, t0, tf):
        """
        Evaluate second term of Magnus expansion over specified time
        interval using pre-calculated terms.
        """
        total = 0
        def x(x): return x

        for coeff in self.H2:
            c = scipy.integrate.dblquad(coeff, t0, tf, t0, x)[0]
            total = total + c*self.H2[coeff]

        return total

    def lvn_solve(self, rho0, tlist):
        """
        Liouville-von Neumann evolution of density matrix for given 
        Hamiltonian.

        Parameters
        ----------
        rho0 : ndarray
            Initial density matrix.
        tlist : list / array
            Times at which to evaluate density matrices.

        Returns
        -------
        ndarray
            Density matrices calculated across tlist.

        Example
        -------
        >>> H = {f : mp.HOp(2,1,mp.sigmax())}
        >>> q_sys = mp.System(H)
        >>> rho0 = mp.HOp(2,1,mp.sigmay())
        >>> tlist = mp.linspace(0, 10, 0.5**5)
        >>> q_sys.lvn_solve(rho0, tlist)

        """
        states = [vec(rho0())]

        for i in range(len(tlist) - 1):

            omega = (self.__eval_first_term(tlist[i], tlist[i+1])
                     + self.__eval_second_term(tlist[i], tlist[i+1]))

            states.append(scipy.linalg.expm(omega) @ states[i])
            states[i] = unvec(states[i])
        
        states[-1] = unvec(states[-1])

        return states