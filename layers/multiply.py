import torch
import torch.nn as nn




class ComplexMultiplyPool(nn.Module):
    def __init__(self,opt):
        super(ComplexMultiplyPool, self).__init__()
        self.opt = opt


    def forward(self, amplitude, phase):
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        real_part = amplitude * cos_phase
        imag_part = amplitude * sin_phase

        real_part = real_part[:, 0]
        imag_part = imag_part[:, 0]

        return real_part, imag_part


class ComplexMultiply(nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, amplitude, phase):
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        real_part = amplitude * cos_phase
        imag_part = amplitude * sin_phase
        return real_part, imag_part
