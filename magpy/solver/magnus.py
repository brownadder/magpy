"""The Magnus expansion for the solution to the Liouville-von Neumann equation.

TODO: Flesh this out with implementation details.

References
----------

.. [1] Magnus, W. (1954), "On the exponential solution of differential 
       equations for a linear operator", *Comm. Pure Appl. Math.* 7, 649-673.

.. [2] Iserles, A., Munthe-Kaas, H. Z., Nørsett, S. P. & Zanna, A. (2000), 
       "Lie-group methods", *Acta Numerica* 9, 215-365.
"""

import itertools
import torch
from math import sqrt

# GL quadrature, degree 3.
knots = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128).reshape((1, 1, 3)) 

# Pairs of knots at which to evaluate commutator of H for second term quadrature. See Iserles, 2000.
knot_slice_indices = itertools.combinations(range(3), 2) 

weights_first_term = torch.tensor([5/9, 8/9, 5/9])
weights_second_term = torch.tensor([2,1,2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
weights_second_term_coeff = sqrt(15) / 54

def batch_first_term(H, tlist, n_qubits):
    """The first term of the Magnus expansion, evaluated across each given
    time interval.

    If `H` is an n-qubit Hamiltonian (shape = [2^n, 2^n]) and `tlist` contains 
    m intervals (m+1 points in time), then this function returns a Tensor with
    shape = [m, 2^n, 2^n].

    Parameters
    ----------
    H : HamiltonianOperator
        System Hamiltonian
    tlist : Tensor
        Discretisation of time
    n_qubits : int
        Number of qubits in system

    Returns
    -------
    Tensor
        Batch first term values
    """

    t0 = tlist[0]
    n = len(tlist) - 1
    step = tlist[1] - tlist[0]

    z = 0.5*step*knots.expand(n, -1, -1) + (t0 + step*(torch.arange(n) + 0.5)).reshape((n, 1, 1)).expand(-1, -1, 3)
    foo = tuple(torch.ones(z.shape)*weights_first_term if f == 1 else f(z)*weights_first_term for f in H.funcs())
    a = 0.5 * step * torch.sum(torch.cat(foo, 1), 2)

    return torch.tensordot(a, torch.stack([p(n_qubits) for p in H.pauli_operators()]), 1)


def batch_second_term(H, tlist, n_qubits):
    """The second term of the Magnus expansion, evaluated across each given 
    time interval.

    If `H` is an n-qubit Hamiltonian (shape = [2^n, 2^n]) and `tlist` contains 
    m intervals (m+1 points in time), then this function returns a Tensor with
    shape = [m, 2^n, 2^n].

    Parameters
    ----------
    H : HamiltonianOperator
        System Hamiltonian
    tlist : Tensor
        Discretisation of time
    n_qubits : int
        Number of qubits in system

    Returns
    -------
    Tensor
        Batch second term values
    """

    n = len(tlist) - 1
    commutators = torch.stack([__eval_commutator(H, tlist, i, j, n, n_qubits) for i, j in knot_slice_indices])
    
    return weights_second_term_coeff * torch.sum(commutators * weights_second_term, 0)


def __eval_commutator(H, tlist, i, j, n, n_qubits):
    # Evaluate the commutator of H at slices i and j of the GL knots over n intervals.

    t0 = tlist[0]
    step = tlist[1] - tlist[0]
    funcs = H.funcs()

    z = (0.5*step*knots.expand(n, -1, -1) 
         + (t0 + step*(torch.arange(n) + 0.5)).reshape((n, 1, 1)).expand(-1, -1, 3)).squeeze()
    z_slice = torch.stack((z[:,i],z[:,j])).transpose(0, 1)

    s = torch.stack([p(n_qubits) for p in H.pauli_operators()])
    f_vals = torch.tensor([[[1 if f == 1 else f(knot) for f in funcs] for knot in knots] for knots in z_slice])
    f_vals_outer_prod = torch.func.vmap(lambda p : torch.outer(p[0], p[1]))(f_vals).unsqueeze(-1).unsqueeze(-1)
    s_outer_prod = torch.einsum('aij,bjk->abik', s, s)

    return (step**2)*torch.sum((f_vals_outer_prod * (s_outer_prod - s_outer_prod.transpose(0, 1)))
                               .reshape((n, len(funcs)**2, 2**n_qubits, 2**n_qubits)), 1)
