from abc import ABC, abstractmethod
import torch
import magpy as mp

class System(ABC):
    """A quantum system"""

    @staticmethod
    def create(H, rho0, tlist):
        """Factory function for instantiating a quantum system.

        Parameters
        ----------
        H : HamiltonianOperator
            System Hamiltonian
        rho0 : PauliString
            Initial density matrix
        tlist : list
            List of time

        Returns
        -------
        System
            An instance representing the quantum system defined
        """

        if isinstance(H, mp.PauliString):
            H = mp.HamiltonianOperator([1, H])

        if H.is_constant():
            return _TimeIndependentQuantumSystem(H, rho0, tlist)
        return _TimeDependentQuantumSystem(H, rho0, tlist)

    @abstractmethod
    def evolve(self):
        """Evolve the density matrix under the specified Hamiltonian."""


class _TimeIndependentQuantumSystem(System):
    def __init__(self, H, rho0, tlist):
        self.H = H
        self.rho0 = rho0
        self.tlist = tlist
        self.n_qubits = max(max(p.qubits.keys()) for p in H.pauli_operators())
        self.states = torch.empty((len(self.tlist), 2**self.n_qubits, 2**self.n_qubits), dtype=torch.complex128)
        self.states[0] = self.rho0(self.n_qubits)

    def evolve(self):
        step = self.tlist[1] - self.tlist[0]

        u = torch.matrix_exp(-1j * step * self.H())
        ut = torch.conj(torch.transpose(u, 0, 1))

        for i in range(len(self.tlist) - 1):
            self.states[i + 1] = u @ self.states[i] @ ut


class _TimeDependentQuantumSystem(System):
    def __init__(self, H, rho0, tlist):
        self.H = H
        self.rho0 = rho0
        self.tlist = tlist
        self.n_qubits = max(max(p.qubits.keys()) for p in H.pauli_operators())
        self.states = torch.empty((len(self.tlist), 2**self.n_qubits, 2**self.n_qubits), dtype=torch.complex128)
        self.states[0] = self.rho0(self.n_qubits)

    def evolve(self):
        omega1 = mp.solver.batch_first_term(self.H, self.tlist, self.n_qubits)
        omega2 = mp.solver.batch_second_term(self.H, self.tlist, self.n_qubits)

        u = torch.matrix_exp(-1j * (omega1 + omega2))
        ut = torch.conj(torch.transpose(u, 1, 2))

        for i in range(len(self.tlist) - 1):
            self.states[i + 1] = u[i] @ self.states[i] @ ut[i]
