# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import copy
import numpy as np
from layers.embedding import PositionEmbedding
from layers.multiply import ComplexMultiply
from layers.evolution import ComplexEvolution
from layers.composite import ComplexComposite
from layers.measurement import ComplexMeasurement

class Pooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, amplitude , phase):
        first_token_amplitude = amplitude[:, 0]
        first_token_phase = phase[:, 0]
        return first_token_amplitude, first_token_phase

class QFM_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(QFM_BERT, self).__init__()
        self.bert = bert
        self.opt = opt
        self.gdense = nn.Linear(opt.embed_dim, opt.sub_dim)
        self.ldense = nn.Linear(opt.embed_dim, opt.sub_dim)
        self.dropout = nn.Dropout(opt.dropout)
        self.position_embedding = PositionEmbedding(self.opt, self.bert.embeddings.word_embeddings.num_embeddings,
                                                    opt.sub_dim)
        self.complex_multiply = ComplexMultiply()
        self.global_evolution = ComplexEvolution(opt)
        self.local_evolution = ComplexEvolution(opt)
        self.complex_composite = ComplexComposite(opt)
        self.complex_measurement = ComplexMeasurement(int(self.opt.sub_dim * self.opt.sub_dim / self.opt.head_num),
                                                      opt.polarities_dim)
        self.pooler = Pooler(self.opt.sub_dim)

    def get_params(self):
        unitary_params = []
        remaining_params = []
        bert_params = []
        bert_params.extend(list(self.bert.parameters()))
        remaining_params.extend(list(self.position_embedding.parameters()))
        remaining_params.extend(list(self.gdense.parameters()))
        remaining_params.extend(list(self.ldense.parameters()))
        remaining_params.extend(list(self.complex_composite.parameters()))
        remaining_params.extend(list(self.complex_measurement.parameters()))

        unitary_params.extend(list(self.global_evolution.parameters()))
        unitary_params.extend(list(self.local_evolution.parameters()))


        return unitary_params, remaining_params,bert_params

    def forward(self, inputs):
        global_context_indices = inputs[0]
        global_context_segments_ids = inputs[1]
        local_context_indices = inputs[2]

        global_amplitude = self.bert(global_context_indices, token_type_ids=global_context_segments_ids)[
            'last_hidden_state']

        global_amplitude = self.dropout(global_amplitude)
        global_amplitude = self.gdense(global_amplitude)
        global_phase = self.position_embedding(global_context_indices)

        local_amplitude = self.bert(local_context_indices)['last_hidden_state']
        local_amplitude = self.ldense(local_amplitude)
        local_amplitude = self.dropout(local_amplitude)
        local_phase = self.position_embedding(local_context_indices)

        global_amplitude,global_phase = self.pooler(global_amplitude,global_phase)
        local_amplitude,local_phase = self.pooler(local_amplitude,local_phase)

        global_real_part, global_imag_part = self.complex_multiply(global_amplitude, global_phase)
        local_real_part, local_imag_part = self.complex_multiply(local_amplitude, local_phase)

        global_real_part, global_imag_part = self.global_evolution(global_real_part, global_imag_part)
        local_real_part, local_imag_part = self.local_evolution(local_real_part, local_imag_part)

        mix_state_real, mix_state_imag = self.complex_composite(global_real_part, global_imag_part, local_real_part,
                                                                local_imag_part)

        output = self.complex_measurement(mix_state_real, mix_state_imag)

        return output
