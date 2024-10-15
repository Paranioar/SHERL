import math
import random

import numpy as np

import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig

from lib.utils import l1norm, l2norm

import logging
logger = logging.getLogger(__name__)


class Downsample(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(input_dim, output_dim),
                                nn.LayerNorm(output_dim))

    def forward(self, x):

        return self.fc(x)


class Guidance(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.fc_attn = nn.Linear(hidden_dim, 1)

    def get_redundancy(self, features):

        norm_features = l2norm(features, dim=-1)
        similarity_map = torch.matmul(norm_features, norm_features.permute(0, 2, 1))
        redundancy = torch.sum(torch.relu(similarity_map), dim=-1, keepdim=True)
        return redundancy
    
    def forward(self, guide_features, all_features):
        
        redundancy = self.get_redundancy(all_features)
        cross_attn = self.fc_attn(all_features * guide_features)
        cross_attn = l1norm(torch.relu(cross_attn) / redundancy, dim=1)

        agg_features = torch.sum(cross_attn * all_features, dim=1, keepdim=True)

        return agg_features + guide_features
    

class sherl_for_transformer(nn.Module):
    def __init__(self, num_layer=12, 
                       input_dim=768, 
                       embed_size=1024, 
                       downsample_factor=2):
        super().__init__()

        hidden_dim = embed_size // downsample_factor
        self.num_layer = num_layer
        self.guide_block = Guidance(input_dim, hidden_dim)

        self.hor_block = nn.ModuleList(
            [Downsample(input_dim, hidden_dim, hidden_dim) for _ in range(num_layer)])

        self.gate_T = 0.1
        self.gate_params = nn.Parameter(torch.ones(1) * 0)
        self.fc_out = nn.Sequential(nn.ReLU(),
                                    nn.Linear(hidden_dim, input_dim, bias=True),
                                    nn.LayerNorm(input_dim, eps=1e-12, elementwise_affine=True),
                                    nn.Dropout(p=0.1, inplace=False))

    def forward(self, all_features_list, last_feature):

        assert self.num_layer == len(all_features_list)
        
        new_features_list = []
        for i, features in enumerate(all_features_list):
            new_features_list.append(self.hor_block[i](features))

        bs, n_instance, dim = new_features_list[0].shape

        shallow_xs = torch.stack(new_features_list[:-1], dim=2).reshape(bs * n_instance, self.num_layer - 1, dim)
        deep_x = new_features_list[-1].reshape(bs * n_instance, 1, dim)

        guide_x = self.guide_block(deep_x, shallow_xs)
        guide_x = guide_x.squeeze(1).reshape(bs, n_instance, -1)

        gate = torch.tanh(self.gate_params / self.gate_T)
        out_x = self.fc_out(guide_x) + gate * last_feature

        return out_x