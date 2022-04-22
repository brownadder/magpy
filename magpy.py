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

    # odd number of elements
    if (len(vec) % 2 != 0):
        if (len(vec) == 1):
            return vec
        else:
            print("Error: odd number of elements in vector. \
                  Cannot form matrix.")
            return None
    elif c is None:
        # matrix is square
        if (np.sqrt(len(vec)).is_integer()):
            c = int(np.sqrt(len(vec)))
        else:
            print("Error: vector cannot form a square matrix. \
                  Please provide a column length, c.")
            return None
    # c does not divide length of vec
    elif (not (len(vec) / c).is_integer()):
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

    # input of the form [a,b,...]
    if len(args) == 1 and isinstance(args[0], list):
        mlist = args[0]
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        # single
        if len(args[0].shape) == 2:
            return args[0]
        # ndarray
        else:
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
        The type of the output array. If dtype is not given,
        then the data type is inferred from arguments. The default is None.

    Returns
    -------
    np.ndarray
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

    # a is an array
    try:
        t = []
        for x in a:
            t.append(np.trace(x.conj().T @ b))
        return np.asarray(t)

    # a is single
    except:
        return np.trace(a.conj().T @ b)


def _magnus_first_term(H_coeffs, HJ, t0, tf):
    omega1 = (tf - t0) * HJ
    for j in range(len(H_coeffs)):
        Ijx = [np.eye(2) for _ in H_coeffs]
        Ijy = [np.eye(2) for _ in H_coeffs]
        Ijz = [np.eye(2) for _ in H_coeffs]
        Ijx[j] = sigmax()
        Ijy[j] = sigmay()
        Ijz[j] = sigmaz()

        omega1 = (omega1
                  + scipy.integrate.quad(H_coeffs[j][0], t0, tf)[0]*kron(Ijx)
                  + scipy.integrate.quad(H_coeffs[j][1], t0, tf)[0]*kron(Ijy)
                  + H_coeffs[j][2]*(tf - t0)*kron(Ijz))

    return liouvillian(omega1)


def _magnus_second_term(H_coeffs, HJ, t0, tf):
    omega2 = 0
    for j in range(len(H_coeffs)):
        Ijx = [np.eye(2) for _ in H_coeffs]
        Ijy = [np.eye(2) for _ in H_coeffs]
        Ijz = [np.eye(2) for _ in H_coeffs]
        Ijx[j] = sigmax()
        Ijy[j] = sigmay()
        Ijz[j] = sigmaz()

        c1 = 2j*H_coeffs[j][2]*kron(Ijx) + commutator(kron(Ijy), HJ)
        c2 = 2j*H_coeffs[j][2]*kron(Ijy) + commutator(HJ, kron(Ijx))
        c3 = 2j * kron(Ijz)

        f = H_coeffs[j][0]
        g = H_coeffs[j][1]

        def x(x): return x
        def q1(y, x): return g(y) - g(x)
        def q2(y, x): return f(y) - f(x)
        def q3(y, x): return f(y)*g(x) - g(y)*f(x)

        int1 = scipy.integrate.dblquad(q1, t0, tf, t0, x)[0]
        int2 = scipy.integrate.dblquad(q2, t0, tf, t0, x)[0]
        int3 = scipy.integrate.dblquad(q3, t0, tf, t0, x)[0]

        omega2 = omega2 + int1*c1 - int2*c2 + int3*c3

    return 0.5j * liouvillian(omega2)


def lvnsolve(H_coeffs, rho0, tlist, HJ=None):
    """
    Liouville-von Neumann evolution of density matrix for given Hamiltonian.

    For n particles, the Hamiltonian takes the form:

    sum_{k=1}^{n} Id otimes  ... otimes (f_k(t)*sigmax + g_k(t)*sigmay
    + omega_k*sigmaz) otimes  ... otimes Id,

    where k denotes position in the kronecker product (otimes).

    For one particle the Hamiltonian takes the form:

    f(t)*sigmax + g(t)*sigmay + omega*sigmaz.

    H_coeffs then takes the form [[f1, g1, omega1], [f2, g2, omega2], ...],
    or [f, g, omega] for a single particle.

    f and g must be functions and the omegas are scalar constants.

    Parameters
    ----------
    H_coeffs : list / array
        Coefficients that form Hamiltonian.
    rho0 : ndarray
        Initial density matrix.
    tlist : list / array
        Times at which to calculate density matrices.
    HJ : ndarray, optional
        Interacting part of Hamiltonian, by default None.

    Returns
    -------
    numpy.ndarray
        Density matrices calculated across tlist.

    Examples
    --------
    One particle :

        >>> def f(t): return t
        >>> def g(t): return t - t**2
        >>> omega = 2
        >>> H_coeffs = [f, g, omega]

    Two particles:

        >>> def f1(t): return t
        >>> def g1(t): return t**2
        >>> omega1 = 2
        >>> def f2(t): return 4*t
        >>> def g2(t): return np.sqrt(t)
        >>> omega2 = -1
        >>> H_coeffs = [[f1, g1, omega1], [f2, g2, omega2]]

    """

    # check whether H_coeffs is a single particle
    # if so convert to list containing only that particle's data
    if not isinstance(H_coeffs[0], (list, np.ndarray)):
        H_coeffs = [H_coeffs]

    # check whether HJ is empty and if it is needed to be
    # compatible with dimension of Hamiltonian
    if HJ is None:
        n = len(H_coeffs)
        HJ = np.zeros((2**n, 2**n))

    states = [vec(rho0)]
    for i in range(len(tlist) - 1):
        omega = (_magnus_first_term(H_coeffs, HJ, tlist[i], tlist[i+1])
                 + _magnus_second_term(H_coeffs, HJ, tlist[i], tlist[i+1]))
        states.append(scipy.linalg.expm(omega) @ states[i])
        states[i] = unvec(states[i])

    states[-1] = unvec(states[-1])

    return states
