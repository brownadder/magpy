import torch
from scipy.integrate import quad, dblquad
from abc import ABC, abstractmethod
import itertools
from magpy.core import commutator
from numbers import Number


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
        self.n_qubits = max(max(n.qubits.keys()) for n in H.data.values())
        self.states = torch.empty((len(self.tlist), 2**self.n_qubits, 2**self.n_qubits), dtype=torch.complex128)
        self.states[0] = self.rho0(self.n_qubits)

    def evolve(self):
        self.states = [self.rho0(n=self.n_qubits)]

        timestep = self.tlist[1] - self.tlist[0]
        u = torch.matrix_exp(-1j * timestep * self.H())
        ut = torch.conj(torch.transpose(1j * timestep * self.H(n=self.n_qubits), 0, 1))

        for i in range(len(self.tlist) - 1):
            self.states[i + 1] = u @ self.states[i] @ ut

        self.states = torch.stack(self.states)


class _TimeDependentQuantumSystem(System):
    def __init__(self, H, rho0, tlist):
        self.H = H
        self.rho0 = rho0
        self.tlist = tlist
        self.n_qubits = max(max(n.qubits.keys()) for n in H.data.values())
        self.states = torch.empty((len(self.tlist), 2**self.n_qubits, 2**self.n_qubits), dtype=torch.complex128)
        self.states[0] = self.rho0(self.n_qubits)

    def evolve(self):
        unpacked = self.__unpack_data()

        for i in range(len(self.tlist) - 1):
            omega1 = self.__first_term(self.tlist[i], self.tlist[i+1], unpacked)
            omega2 = -0.5j * self.__second_term(self.tlist[i], self.tlist[i+1], unpacked)

            u = torch.matrix_exp(-1j * (omega1 + omega2))
            ut = torch.conj(torch.transpose(u, 0, 1))

            self.states[i + 1] = u @ self.states[i] @ ut

    def __first_term(self, t0, tf, unpacked):
        return sum(
            (f*(tf - t0) if isinstance(f, Number)
                else quad(f, t0, tf)[0]) * h(self.n_qubits) for f, h in unpacked)

    def __second_term(self, t0, tf, unpacked):
        total = 0

        for i, j in itertools.permutations(range(len(unpacked)), 2):
            com = commutator(unpacked[i][1], unpacked[j][1])

            if (com.scale != 0):
                c = dblquad(lambda y, x: unpacked[i][0](x) * unpacked[j][0](y), t0, tf, t0, lambda x: x)[0]
                total += c * com(self.n_qubits)

        return total

    def __unpack_data(self):
        return [(k, v) for k, items in self.H.data.items() for v in (items if isinstance(items, list) else [items])]
