import torch
import torch.nn as nn
from torch import tanh
from .measurement import QMatrix

def complex_tanh(input_r, input_i):
    return tanh(input_r), tanh(input_i)


class ComplexPooler(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dense = QMatrix(opt.embed_dim, opt.embed_dim)

    def forward(self, hidden_states_r, hidden_states_i):
        first_token_tensor_r = hidden_states_r[:, 0]
        first_token_tensor_i = hidden_states_i[:, 0]
        pooled_output_r, pooled_output_i = self.dense(first_token_tensor_r, first_token_tensor_i)
        pooled_output_r, pooled_output_i = complex_tanh(pooled_output_r, pooled_output_i)
        return pooled_output_r, pooled_output_i