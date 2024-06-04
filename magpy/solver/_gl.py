from math import sqrt
import itertools
import torch
from .._device import _DEVICE_CONTEXT

# GL quadrature, degree 3.
knots = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128).reshape((1, 1, 3)).to(_DEVICE_CONTEXT.device)

# Pairs of knots at which to evaluate commutator of H for second term quadrature. See Iserles, 2000.
knot_slice_indices = itertools.combinations(range(3), 2)

weights_first_term = torch.tensor([5/9, 8/9, 5/9]).to(_DEVICE_CONTEXT.device)
weights_second_term = torch.tensor([2,1,2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(_DEVICE_CONTEXT.device)
weights_second_term_coeff = sqrt(15) / 54
