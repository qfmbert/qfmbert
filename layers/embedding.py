import torch
import torch.nn as nn
import numpy as np


class PositionEmbedding(nn.Module):
    def __init__(self, opt,input_dim,embed_dim):
        super(PositionEmbedding, self).__init__()
        self.opt = opt
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim), embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix)

        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix)

    def forward(self,x):
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)

        time_stamps = x.shape[1]

        positions = torch.arange(time_stamps).unsqueeze(-1).to(self.opt.device)
        pos_embed = positions.repeat(1, self.embed_dim) * self.frequency_embedding(x) + phases

        return pos_embed


class ZeroEmbedding(nn.Module):
    def __init__(self, opt,input_dim,embed_dim):
        super(ZeroEmbedding, self).__init__()
        self.opt = opt
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.phase_embedding = nn.Embedding.from_pretrained(torch.zeros([self.input_dim,self.embed_dim]))


    def forward(self,x):
        pos_embed = self.phase_embedding(x)
        return pos_embed