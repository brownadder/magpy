import magpy as mp
from numbers import Number


class PauliString:
    """A multi-qubit Pauli operator.

    A representation of a tensor product of single qubit Pauli operators.
    Identity operators are inferred from the gaps in the indices in the
    internal dictionary.

    Attributes
    ----------
    qubits : dict
        The qubits and their indices in the operator
    scale : Number
        Scalar coefficient
    """

    def __init__(self, x=None, y=None, z=None, scale=1):
        """A multi-qubit Pauli operator.

        Parameters
        ----------
        x, y, z : set[int], optional
            Position of the Pauli qubits in the operator, by default None
        scale : int, optional
            Scalar coefficient, by default 1
        """

        self.qubits = {}
        self.scale = scale

        for q, label in zip([x, y, z], ["x", "y", "z"]):
            try:
                self.qubits |= {n: label for n in q}
            except TypeError:
                if q is not None:
                    self.qubits[q] = label

    def __eq__(self, other):
        return self.qubits == other.qubits and self.scale == other.scale

    def __mul__(self, other):
        if isinstance(other, mp.HamiltonianOperator):
            return -other * self

        s = PauliString(scale=0)

        try:
            if other.scale == 0:
                return s

            s.qubits = self.qubits | other.qubits
        except AttributeError:
            if isinstance(other, Number):
                if other == 0:
                    return s

                s.scale = self.scale * other
                s.qubits = self.qubits
            else:
                try:
                    # other is FunctionProduct.
                    self *= other.scale
                    other.scale = 1
                except AttributeError:
                    # other is other type of function.
                    pass

                return mp.HamiltonianOperator([other, self])

        else:
            # other is PauliString.
            s.scale = self.scale * other.scale

            for n in list(set(self.qubits.keys() & other.qubits.keys())):
                if self.qubits[n] == other.qubits[n]:
                    del s.qubits[n]
                else:
                    scale, spin = PauliString.__pauli_mul(self.qubits[n], other.qubits[n])
                    s.scale *= 1j * scale
                    s.qubits[n] = spin

        return s

    __rmul__ = __mul__

    def __add__(self, other):
        if self == other:
            s = PauliString(scale=self.scale + other.scale)
            s.qubits = self.qubits
            return s
        return mp.HamiltonianOperator([1, self], [1, other])

    def __neg__(self):
        s = PauliString(scale=-self.scale)
        s.qubits = self.qubits
        return s

    def __repr__(self):
        return str(self.scale) + "*" + str(dict(sorted(self.qubits.items()))).replace("'", "")

    @staticmethod
    def X(*args):
        """Multi-qubit operator formed of Pauli X operators."""

        return PauliString(x=args if args else 1)

    @staticmethod
    def Y(*args):
        """Multi-qubit operator formed of Pauli Y operators."""
        return PauliString(y=args if args else 1)

    @staticmethod
    def Z(*args):
        """Multi-qubit operator formed of Pauli Z operators."""
        return PauliString(z=args if args else 1)

    @staticmethod
    def Id():
        """The identity operator."""
        return PauliString()

    @staticmethod
    def collect(arr):
        """Group PauliStrings in list which have the same qubit structure.

        Parameters
        ----------
        arr : list[PauliString]
            PauliString instances

        Returns
        -------
        list[PauliString]|PauliString
            Collected instances or single instance
        """

        counts = {}  # Number of occurrences of each unique PauliString.
        out = []

        try:
            for ps in arr:
                scale = ps.scale
                ps = tuple(ps.qubits.items())

                try:
                    counts[ps] += scale
                except KeyError:
                    counts[ps] = scale
        except TypeError:  # arr is single PauliString.
            return arr
        else:
            for c in counts:
                a = PauliString()
                a.qubits = dict(c)
                a.scale = counts[c]
                out.append(a)

        return out[0] if len(out) == 1 else out

    @staticmethod
    def __e_ijk(i, j, k):
        # Levi-Civita symbol.
        return int((i - j) * (j - k) * (k - i) / 2)

    @staticmethod
    def __pauli_mul(a, b):
        # Composition of two Pauli qubits.
        if a == b:
            return None
        c = "xyz".replace(a, "").replace(b, "")
        return PauliString.__e_ijk(ord(a), ord(b), ord(c)), c
