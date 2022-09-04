import torch
import torch.nn as nn


class QMatrix(nn.Module):
    def __init__(self, in_features, out_features):
        super(QMatrix, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        output_r = self.fc_r(input_r) - self.fc_i(input_i)
        output_i = self.fc_r(input_i) + self.fc_i(input_r)
        return output_r, output_i


class ComplexMeasurement(nn.Module):
    def __init__(self, emb_dim, output_dim):
        super(ComplexMeasurement, self).__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.M_t = QMatrix(self.emb_dim, self.output_dim)
        self.M = QMatrix(self.emb_dim, self.output_dim)

    def forward(self, inputs_r, input_i):
        left_r, left_i = self.M_t(inputs_r, input_i)
        right_r, right_i = self.M(inputs_r, input_i)
        output_r = left_r * right_r - left_i * right_i
        output_i = left_i * right_r + left_r * right_i
        output = torch.sqrt(torch.pow(output_r, 2) + torch.pow(output_i, 2))
        return output


class Measurement(nn.Module):
    def __init__(self, emb_dim, output_dim):
        super(Measurement, self).__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.M = nn.Linear(self.emb_dim, self.output_dim)
        self.M_t = nn.Linear(self.emb_dim, self.output_dim)

    def forward(self, input):
        output = self.M(input)
        output_t = self.M_t(input)
        out = output * output_t
        return out
