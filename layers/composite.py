import torch
import torch.nn as nn


class ComplexComposite(nn.Module):
    def __init__(self, opt):
        super(ComplexComposite, self).__init__()
        self.opt = opt

    def complex_kron(self, a_r, a_i, b_r, b_i):
        c_r = self.kron(a_r, b_r) - self.kron(a_i, b_i)
        c_i = self.kron(a_r, b_i) + self.kron(a_i, b_r)
        return c_r, c_i



    def kron(self, a, b):
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]

        c = res.reshape(siz0 + siz1)
        c = c.squeeze()
        return c

    def forward(self, global_real_part, global_imag_part, local_real_part,
                local_imag_part):

        global_real_part = torch.chunk(global_real_part, self.opt.head_num, dim=1)
        global_imag_part = torch.chunk(global_imag_part, self.opt.head_num, dim=1)
        local_real_part = torch.chunk(local_real_part, self.opt.head_num, dim=1)
        local_imag_part = torch.chunk(local_imag_part, self.opt.head_num, dim=1)


        global_real_part = torch.stack(global_real_part)
        global_imag_part = torch.stack(global_imag_part)
        local_real_part = torch.stack(local_real_part)
        local_imag_part = torch.stack(local_imag_part)

        mix_real_list = []
        mix_imag_list = []
        for item in zip(global_real_part, global_imag_part, local_real_part, local_imag_part):
            g_r = item[0]
            g_i = item[1]
            l_r = item[2]
            l_i = item[3]
            mix_r, mix_i = self.complex_kron(g_r,g_i,l_r,l_i)
            mix_real_list.append(mix_r)
            mix_imag_list.append(mix_i)

        mix_real_list = torch.stack(mix_real_list)
        mix_imag_list = torch.stack(mix_imag_list)
        mix_real_list = mix_real_list.permute(1, 0, 2)

        mix_real_list = mix_real_list.reshape(mix_real_list.shape[0],
                                                               mix_real_list.shape[1]*
                                                               mix_real_list.shape[2])
        mix_imag_list = mix_imag_list.permute(1, 0, 2)

        mix_imag_list = mix_imag_list.reshape(mix_imag_list.shape[0],
                                                               mix_imag_list.shape[1] *
                                                               mix_imag_list.shape[2])

        return mix_real_list, mix_imag_list
