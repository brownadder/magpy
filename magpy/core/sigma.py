class Sigma():
    """
    An n-spin quantum operator formed of Pauli matrices.

    The representation of the operator is reduced to a dictionary of positions
    in a Kronecker product and a combination of Pauli matrices at each.

    Any unspecified positions in the product are inferred to be identity, and 
    the actual number of spins on which the operator acts is not stored.

    Attributes
    ----------
    spins : dict
        Position in product and their respective matrices.
    
    Examples
    --------
    sigmax acting on the first spin:
        >>> Sigma(x = 1)

    sigmax acting on the first spin and sigmay on the second:
        >>> Sigma(x = 1, y = 2)

    sigmax acting on the first and third spins:
        >>> Sigma(x = {1, 3})

        The internal representation of this operator is {1 : 'x', 3 : 'x'}.

    """  

    def __init__(self, x=None, y=None, z=None):
        """
        Construct an operator with the given positions and matrices.

        Parameters
        ----------
        x : int or set of int, optional
            Position(s) of the Pauli x matrix, by default None
        y : int or set of int, optional
            Position(s) of the Pauli y matrix, by default None
        z : int or set of int, optional
            Position(s) of the Pauli z matrix, by default None

        """
        
        self.spins = {}
        for spin, label in zip([x, y, z], ["x", "y", "z"]):
            try:
                self.spins |= dict([(n, label) for n in spin])
            except:
                if spin is not None:
                    self.spins[spin] = label
    
    def __mul__(self, other):
        """
        Compose two quantum operators.

        Parameters
        ----------
        other : Sigma
            A quantum operator.

        Returns
        -------
        Sigma
            The resultant quantum operator from the composition.

        Examples
        --------

        >>> (Sigma(x=1) * Sigma(y=2)).spins
        {1: 'x', 2: 'y'}

        >>> (Sigma(x=1) * Sigma(z=1)).spins
        {1 : 'xz'}

        """
        left = self.spins.keys()
        right = other.spins.keys()
        overlap = [n for n in left if n in right]

        s = Sigma()
        s.spins = self.spins | other.spins \
            | dict([(n, self.spins[n] + other.spins[n]) for n in overlap])

        return s