import torch


def complex_abs(x):
    xr = x[..., 0]**2
    xi = x[..., 1]**2
    return xr + xi


def complex_product(x, y):
    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack([real, imag], -1)
