import random

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils import Flatten

import pytorch_lightning as pl


def apply_layer(real_function, x):
    psi = real_function(x.real) + 1j * real_function(x.imag)
    m_psi = psi.abs()
    phi_psi = stable_angle(psi)

    chi = real_function(x.abs())
    m_psi = 0.5 * m_psi + 0.5 * chi

    return m_psi, phi_psi


def stable_angle(x: torch.tensor, eps=1e-8):
    """ Function to ensure that the gradients of .angle() are well behaved."""
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return y.angle()


def apply_activation_function(m, phi, channel_norm):
    m = channel_norm(m)
    m = torch.nn.functional.relu(m)
    return get_complex_number(m, phi)


def get_complex_number(magnitude, phase):
    return magnitude * torch.exp(phase * 1j)

def complex_addition(z1, z2):
    return (z1.real + z2.real) + 1j*(z1.imag + z2.imag)

def complex_multiplication(z1, z2):
    return (z1.abs() * z2.abs()) * torch.exp((z1.angle()+z2.angle())/2 * 1j)

class ComplexLinear(pl.LightningModule):
    def __init__(self, in_dim, out_dim, biases, last=False):
        super(ComplexLinear, self).__init__()
        self.weights = nn.Linear(in_dim, out_dim, bias=biases)
        self.norm = nn.LayerNorm(out_dim)
        self.last = last
        if last:
            self.threshold = torch.Tensor([0.6])

    def forward(self, z_in):
        z_out = apply_layer(self.weights, z_in)
        if self.last:
            self.threshold = (torch.max(z_out[0], dim=1).values)/3
            pred = z_out[0] - self.threshold[:,None]#.to(self.device)
            return pred, get_complex_number(z_out[0], z_out[1])

        else:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
            pred = z_out
            return pred


class ComplexConvolution(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases):
        super(ComplexConvolution, self).__init__()
        self.padding = padding
        self.channels = channels
        self.kernel_conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)

        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, z_in):
        z_out = apply_layer(self.kernel_conv, z_in)
        z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        return z_out


class ComplexTransposeConvolution(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, output_padding, biases):
        super(ComplexTransposeConvolution, self).__init__()
        self.padding = padding
        self.channels = channels
        self.kernel_conv = nn.ConvTranspose2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                              padding=padding, output_padding=output_padding, bias=biases)

        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, z_in):
        z_out = apply_layer(self.kernel_conv, z_in)
        z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        return z_out


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class ComplexMaxPool2d(pl.LightningModule):
    def __init__(self, kernel_size):
        super(ComplexMaxPool2d, self).__init__()
        self.maxpool = nn.MaxPool2d((kernel_size, kernel_size), return_indices=True, ceil_mode=True)

    def forward(self, z_in):
        m_psi = z_in.abs()
        phi_psi = stable_angle(z_in)
        pooled_mag, indexes = self.maxpool(m_psi)
        pooled_phases = retrieve_elements_from_indices(phi_psi, indexes)
        return get_complex_number(pooled_mag, pooled_phases)


class ComplexClassifier(pl.LightningModule):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.fl = Flatten()
        self.l1 = nn.Linear(channels * 4 * 4, 50)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(50, num_classes)


    def forward(self, z_in):

        amplitude = z_in.abs()
        prediction = self.l2(self.relu(self.l1(self.fl(amplitude))))
        return prediction
