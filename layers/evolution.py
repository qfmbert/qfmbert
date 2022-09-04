import torch
import torch.nn as nn

from layers.measurement import QMatrix
from torch import relu



class ComplexEvolution(nn.Module):
    def __init__(self, opt):
        super(ComplexEvolution, self).__init__()
        self.opt = opt
        self.U = torch.nn.Parameter(torch.stack([torch.eye(self.opt.sub_dim).to(self.opt.device),
                                                 torch.zeros(self.opt.sub_dim, self.opt.sub_dim).to(self.opt.device)],
                                                dim=-1))
    def forward(self, input_r, input_i):
        U_real = self.U[:, :, 0]
        U_imag = self.U[:, :, 1]
        bs = input_r.shape[0]
        U_real = U_real.unsqueeze(0).expand(bs, -1, -1)
        U_imag = U_imag.unsqueeze(0).expand(bs, -1, -1)

        input_r = input_r.unsqueeze(-1)
        input_i = input_i.unsqueeze(-1)
        r = torch.matmul(U_real, input_r) - torch.matmul(U_imag, input_i)
        i = torch.matmul(U_imag, input_r) + torch.matmul(U_real, input_i)
        r = r.squeeze()
        i = i.squeeze()
        return r, i


class Evolution(nn.Module):
    def __init__(self, opt):
        super(Evolution, self).__init__()
        self.opt = opt
        self.U = nn.Linear(self.opt.embed_dim, self.opt.embed_dim)

    def forward(self, input):
        output = self.U(input)
        return output


class UEvolution(nn.Module):
    def __init__(self, opt):
        super(UEvolution, self).__init__()
        self.opt = opt
        self.unitary_x = torch.nn.Parameter(
            torch.stack([torch.eye(self.opt.sub_dim).unsqueeze(0).expand(self.opt.max_seq_len,-1,-1).to(self.opt.device), torch.zeros(self.opt.sub_dim, self.opt.sub_dim).unsqueeze(0).expand(self.opt.max_seq_len,-1,-1).to(self.opt.device)],
                        dim=-1))

    def forward(self, input_r, input_i):
        output_r, output_i = self.evolution(input_r,input_i)

        return output_r, output_i

    def evolution(self,input_r,input_i):
        U_real = self.unitary_x[:, :, :,0]
        U_imag = self.unitary_x[:, :, :,1]
        output_real = []
        output_imag = []
        for x_real, x_imag in zip(input_r, input_i):
            _r = torch.matmul(U_real, x_real.unsqueeze(-1)).squeeze() - torch.matmul(U_imag, x_imag.unsqueeze(-1)).squeeze()
            _i = torch.matmul(U_imag, x_real.unsqueeze(-1)).squeeze() + torch.matmul(U_real, x_imag.unsqueeze(-1)).squeeze()

            output_real.append(_r)
            output_imag.append(_i)
        output_real = torch.stack(output_real, dim=0)
        output_imag = torch.stack(output_imag, dim=0)

        return output_real, output_imag