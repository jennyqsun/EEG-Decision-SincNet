# Created on 9/24/22 at 5:59 PM 

# Author: Jenny Sun

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch import fft
from typing import List, Sequence, Union


class HilbertTransform(nn.Module):
    """
    Determine the analytical signal of a Tensor along a particular axis.
    Args:
        axis: Axis along which to apply Hilbert transform. Default 2 (first spatial dimension).
        n: Number of Fourier components (i.e. FFT size). Default: ``x.shape[axis]``.
    """

    def __init__(self, axis: int = 2, n: Union[int, None] = None) -> None:

        super().__init__()
        self.axis = axis
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to transform. Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
        Returns:
            torch.Tensor: Analytical signal of ``x``, transformed along axis specified in ``self.axis`` using
            FFT of size ``self.N``. The absolute value of ``x_ht`` relates to the envelope of ``x`` along axis ``self.axis``.
        """

        # Make input a real tensor
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError(f"Invalid axis for shape of x, got axis {self.axis} and shape {x.shape}.")

        n = x.shape[self.axis] if self.n is None else self.n
        if n <= 0:
            raise ValueError("N must be positive.")
        x = torch.as_tensor(x, dtype=torch.complex64)
        # Create frequency axis
        f = torch.cat(
            [
                torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
                torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
            ]
        )
        xf = fft.fft(x, n=n, dim=self.axis)
        # Create step function
        u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
        new_dims_before = self.axis
        new_dims_after = len(xf.shape) - self.axis - 1
        for _ in range(new_dims_before):
            u.unsqueeze_(0)
        for _ in range(new_dims_after):
            u.unsqueeze_(-1)

        ht = fft.ifft(xf * 2 * u, dim=self.axis)

        # Apply transform
        return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)




# calculate the envolope
import matplotlib.pyplot as plt
import numpy as np

def makeplots():
    '''TODO: fix the argin and argout'''
    timeV = torch.linspace(0,0.5,1000)
    data = (3*torch.sin(2*math.pi*23*timeV)) * (1+torch.cos(2*math.pi*25*timeV)) * \
           (2+torch.sin(2*math.pi*4*timeV)) * torch.sin(2*math.pi*1*timeV) + torch.cos(2*math.pi*5*timeV)
    # data2 = (1+torch.sin(2*math.pi*28*timeV)) * (torch.cos(2*math.pi*30*timeV))
    # data = torch.stack((data,data2))
    hilbert = HilbertTransform(axis = 0)
    out = hilbert(data)
    plt.figure()
    plt.plot(timeV,data.numpy().T)
    plt.show()
    plt.figure()
    plt.plot(timeV,data.numpy().T)
    plt.plot(timeV, torch.abs(out).numpy().T, color = 'red',linewidth = 5)
    # plt.plot(timeV, torch.abs(out).numpy().T * -1, color= 'green',linewidth = 5)
    plt.show()
    return out