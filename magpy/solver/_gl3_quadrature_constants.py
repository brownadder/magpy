from math import sqrt
import torch
from .._device import _DEVICE_CONTEXT

# GL quadrature, degree 3.
knots = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128).reshape((1, 1, 3)).to(_DEVICE_CONTEXT.device)

weights_first_term = torch.tensor([5/9, 8/9, 5/9]).to(_DEVICE_CONTEXT.device)
weights_second_term = torch.tensor([2,1,2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(_DEVICE_CONTEXT.device)
weights_second_term_coeff = sqrt(15) / 54
