import torch
import scipy.integrate
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


class _TimeDependentQuantumSystem(System):

    def __init__(self, H, rho0, tlist):
        self.H = H
        self.rho0 = rho0
        self.tlist = tlist
        self.states = None

    def evolve(self):
        self.states = [self.rho0()]

        # Draft version of 1-term Magnus.
        temp = []
        for f, h in self.H.data.items():
            if isinstance(h, list):
                for e in h:
                    temp.append((f, e))
            else:
                temp.append((f, h))

        for i in range(len(self.tlist) - 1):
            omega = -1j*sum(scipy.integrate.quad(f, self.tlist[i], self.tlist[i+1])[0] * h() for f, h in temp)
            u = torch.matrix_exp(omega)
            ut = torch.conj(torch.transpose(u, 0, 1))

            self.states.append(u @ self.states[i] @ ut)
