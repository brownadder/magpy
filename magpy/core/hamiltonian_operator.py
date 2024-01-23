import magpy as mp
from copy import deepcopy
from numbers import Number


class HamiltonianOperator:
    """A Hamiltonian operator.

    A representation of a Hamiltonian operator, formed of a sum of Pauli
    operators with functional coefficients.

    Attributes
    ----------
    data : dict
        Functional coefficients and their PauliStrings
    """

    def __init__(self, *pairs):
        """A Hamiltonian operator.

        Parameters
        ----------
        *pairs : tuple
            Pairs of functions and PauliStrings
        """

        self.data = {}

        for pair in pairs:
            try:
                # Move any constant coefficients to the corresponding PauliString.
                if pair[0].scale != 1:
                    pair[1] *= pair[0].scale
                    pair[0].scale = 1
            except AttributeError:
                if isinstance(pair[0], Number):
                    pair[1] *= pair[0]
                    pair[0] = 1

            try:
                self.data[pair[0]].append(pair[1])
            except KeyError:
                self.data[pair[0]] = pair[1]
            except AttributeError:
                self.data[pair[0]] = [self.data[pair[0]], pair[1]]

        HamiltonianOperator.__simplify(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __mul__(self, other):
        if isinstance(other, HamiltonianOperator):
            raise NotImplementedError(
                "HamiltonianOperator composition is not yet implemented")

        out = HamiltonianOperator()
        out.data = deepcopy(self.data)

        if isinstance(other, Number | mp.PauliString):
            for coeff in out.data:
                try:
                    for i in range(len(out.data[coeff])):
                        out.data[coeff][i] *= other
                except TypeError:
                    out.data[coeff] *= other

        else:
            # other is FunctionProduct or other type of function.
            for coeff in list(out.data):
                out.data[mp.FunctionProduct(coeff, other)] = out.data.pop(coeff)

        return out

    __rmul__ = __mul__

    def __add__(self, other):
        out = HamiltonianOperator()

        try:
            out.data = self.data | other.data
        except AttributeError:
            # other is PauliString; add it to constants.
            out.data = self.data.copy()

            try:
                out.data[1].append(other)
            except KeyError:
                out.data[1] = other
            except AttributeError:
                out.data[1] = [out.data[1], other]
        else:
            # other is HamiltionianOperator.
            for coeff in list(set(self.data.keys() & other.data.keys())):
                out.data[coeff] = []

                try:
                    out.data[coeff].extend(self.data[coeff])
                except TypeError:
                    out.data[coeff].append(self.data[coeff])

                try:
                    out.data[coeff].extend(other.data[coeff])
                except TypeError:
                    out.data[coeff].append(other.data[coeff])

        HamiltonianOperator.__simplify(out.data)
        return out

    def __neg__(self):
        out = HamiltonianOperator()
        out.data = deepcopy(self.data)
        return -1 * out

    def __repr__(self):
        return str(self.data)

    @staticmethod
    def __simplify(arrs):
        # Collect all PauliStrings in all lists in arrs.
        for coeff in arrs:
            arrs[coeff] = mp.PauliString.collect(arrs[coeff])
