class Sigma():
    """
    An n-spin quantum operator formed of Pauli matrices.

    The representation of the operator is stored as a dictionary of sites in a 
    Kronecker product and a combination of Pauli matrices at each.

    Any unspecified sites in the product are inferred to be identity, and the 
    actual number of spins on which the operator acts is not stored.

    Attributes
    ----------
    spins : dict
        Sites in product and their respective matrices.
    scale : complex
        Scalar coefficient of operator, by default 1.
    
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

    def __init__(self, x={}, y={}, z={}, scale=1):
        """
        Construct an operator with the given positions and matrices.

        Parameters
        ----------
        x : int or set of int, optional
            Site(s) of the Pauli x matrix, by default {}.
        y : int or set of int, optional
            Site(s) of the Pauli y matrix, by default {}.
        z : int or set of int, optional
            Site(s) of the Pauli z matrix, by default {}.
        """
        
        self.scale = scale
        self.spins = {}

        if scale == 0:
            return

        for spin, label in zip([x, y, z], ['x', 'y', 'z']):
            try:
                self.spins |= dict([(n, label) for n in spin])
            except:
                if spin is not None:
                    self.spins[spin] = label
    

    @staticmethod
    def X(sites=1):
        """
        Return a quantum operator formed of Pauli X matrices.

        Parameters
        ----------
        sites : int or set of int, optional
            Site(s) of Pauli X matrices, by default 1.
        """
        return Sigma(x = sites)
    

    @staticmethod
    def Y(sites=1):
        """
        Return a quantum operator formed of Pauli Y matrices.

        Parameters
        ----------
        sites : int or set of int, optional
            Site(s) of Pauli Y matrices, by default 1.
        """
        return Sigma(y = sites)
    

    @staticmethod
    def Z(sites=1):
        """
        Return a quantum operator formed of Pauli Z matrices.

        Parameters
        ----------
        sites : int or set of int, optional
            Site(s) of Pauli Z matrices, by default 1.
        """
        return Sigma(z = sites)


    def __str__(self):
        """
        Return a pretty presentation of the quantum operator.
        """

        sep = ': ' if self.spins else ''
        return str(self.scale) + sep \
            + ' '.join([str(n) + ':' + s for (n, s) in sorted(self.spins.items())])
    

    def __repr__(self):
        """
        Return the internal representation of the quantum operator's data.
        """
        return str(self.scale) + ": " + str(dict(sorted(self.spins.items())))


    def __eq__(self, other):
        """
        Return true if the operators have the same spins in the same sites.
        """
        return self.spins == other.spins and self.scale == other.scale


    def __neg__(self):
        """
        Negate quantum operator.
        """

        self.scale *= -1
        return self


    @staticmethod
    def __e_ijk(i, j, k):
        """
        Three-dimensional Levi-Civita symbol.
        """
        return (i - j)*(j - k)*(k - i) / 2


    @staticmethod
    def __pauli_mul(a, b):
        """
        Return product of two Pauli matrices (None if identity).

        Parameters
        ----------
        a : str
            'x', 'y', or 'z'
        b : str
            'x', 'y', or 'z'

        Returns
        -------
        complex
            Scalar coefficient of product.
        str
            'x', 'y', 'z', or None
        """

        if a == b:
            # Identity.
            return 1, None
        
        c = 'xyz'.replace(a, '').replace(b, '')
        return Sigma.__e_ijk(ord(a), ord(b), ord(c)), c


    def __mul__(self, other):
        """
        Compose quantum operators.

        Examples
        --------
        >>> 3j * Sigma(x=1) * Sigma(y=2)
        3j: {1: 'x', 2: 'y'}

        >>> Sigma(x=1) * Sigma(z=1)
        -1: {1 : 'y'}
        """

        s = Sigma()
        s.scale = 0

        if self.scale == 0:
            return s
        
        try: 
            # other is Sigma.
            if other.scale == 0:
                return s
            
            s.scale = self.scale * other.scale
            overlap_spins = {}

            for n in [n for n in self.spins.keys() if n in other.spins.keys()]:
                # Scalar coefficient, spin.
                p = Sigma.__pauli_mul(self.spins[n], other.spins[n])

                if (p[1] is None):
                    # Identity matrix, therefore remove.
                    del self.spins[n]
                    del other.spins[n]
                else:
                    overlap_spins[n] = p[1]
                    s.scale *= p[0] * 1j

            s.spins = self.spins | other.spins | overlap_spins
        except: 
            # other is int.
            if other == 0:
                return s
            
            s.scale = self.scale * other
            s.spins = self.spins
        
        return s
    

    def __rmul__(self, other):
        """
        Compose quantum operators.

        Examples
        --------
        >>> 3j * Sigma(x=1) * Sigma(y=2)
        3j: {1: 'x', 2: 'y'}

        >>> Sigma(x=1) * Sigma(z=1)
        -1: {1 : 'y'}
        """
        return self.__mul__(other)