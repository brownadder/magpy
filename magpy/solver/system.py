import torch
from abc import ABC, abstractmethod


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
        self.states = None

    def evolve(self):
        self.states = [self.rho0()]

        timestep = self.tlist[1] - self.tlist[0]
        lhs = torch.matrix_exp(-1j * timestep * self.H())
        rhs = torch.matrix_exp(1j * timestep * self.H())

        for i in range(len(self.tlist) - 1):
            self.states.append(lhs @ self.states[i] @ rhs)

    def show_states(self):
        for state in self.states:
            print(state)
            print("---")
