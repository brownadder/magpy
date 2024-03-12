import torch
import scipy.integrate
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

        unpacked = []
        for f, h in self.H.data.items():
            if isinstance(h, list):
                for e in h:
                    unpacked.append((f, e))
            else:
                unpacked.append((f, h))

        for i in range(len(self.tlist) - 1):
            omega1 = self.__first_term(self.tlist[i], self.tlist[i+1], unpacked)
            omega2 = 0.5 * self.__second_term(self.tlist[i], self.tlist[i+1])
            omega = omega1 + omega2

            u = torch.matrix_exp(-1j * omega)
            ut = torch.conj(torch.transpose(u, 0, 1))
            self.states.append(u @ self.states[i] @ ut)

    def __first_term(self, t0, tf, unpacked):
        return sum(
            (f*(tf - t0) if isinstance(f, Number)
                else scipy.integrate.quad(f, t0, tf)[0]) * h() for f, h in unpacked)

    def __second_term(self, a, b):
        ps = self.__unpack_data()
        pairs = itertools.product(ps, repeat=2)

        out = 0
        for pair in pairs:
            if pair[0][1] == pair[1][1]:
                continue
            out += self.__second_term_integral(pair[0][0], pair[1][0], a, b) * commutator(pair[0][1], pair[1][1])()

        return out

    def __second_term_integral(self, f, g, t0, tf):
        if isinstance(f, Number) and isinstance(g, Number):
            return scipy.integrate.dblquad(lambda y, x: f * g, t0, tf, t0, lambda x: x)[0]

        if isinstance(f, Number):
            return scipy.integrate.dblquad(lambda y, x: f * g(y), t0, tf, t0, lambda x: x)[0]

        if isinstance(g, Number):
            return scipy.integrate.dblquad(lambda y, x: f(x) * g, t0, tf, t0, lambda x: x)[0]

        return scipy.integrate.dblquad(lambda y, x: f(x) * g(y), t0, tf, t0, lambda x: x)[0]

    def __unpack_data(self):
        return [(k, v) for k, items in self.H.data.items() for v in (items if isinstance(items, list) else [items])]
