import torch
import magpy as mp
from .._device import _DEVICE_CONTEXT


class System():
    """A quantum system (pure or mixed) which evolves under the 
    Liouville-von Neumann equation."""

    def __init__(self, H, rho0, tlist):
        """Instantiate a quantum system.

        Parameters
        ----------
        H : HamiltonianOperator
            System Hamiltonian
        rho0 : PauliString
            Initial density matrix
        tlist : list
            List of points in time
        """
        
        self.H = H
        self.rho0 = rho0
        self.tlist = tlist
        self.n_qubits = max(max(p.qubits.keys()) for p in H.pauli_operators())
        self.states = torch.empty((len(self.tlist), 2**self.n_qubits, 2**self.n_qubits), dtype=torch.complex128) \
            .to(_DEVICE_CONTEXT.device)
        self.states[0] = self.rho0(self.n_qubits)

    def evolve(self):
        """Evolve the density matrix under the specified Hamiltonian,
        starting from the initial condition."""

        if self.H.is_constant():
            return _evolve_time_independent(self)
        return _evolve_time_dependent(self)


def _evolve_time_independent(self):
    step = self.tlist[1] - self.tlist[0]

    u = torch.matrix_exp(-1j * step * self.H())
    ut = torch.conj(torch.transpose(u, 0, 1))

    for i in range(len(self.tlist) - 1):
        self.states[i + 1] = u @ self.states[i] @ ut


def _evolve_time_dependent(self):
    omega1 = mp.solver.batch_first_term(self.H, self.tlist, self.n_qubits)
    omega2 = mp.solver.batch_second_term(self.H, self.tlist, self.n_qubits)

    u = torch.matrix_exp(-1j * (omega1 + omega2))
    ut = torch.conj(torch.transpose(u, 1, 2))

    for i in range(len(self.tlist) - 1):
        self.states[i + 1] = u[i] @ self.states[i] @ ut[i]
