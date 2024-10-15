import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig

from lib.utils import l1norm, l2norm, count_params

import logging
logger = logging.getLogger(__name__)


def flat_featuremap(feature):
    assert len(feature.shape) == 4
    return feature.reshape(feature.size(0), feature.size(1), -1).permute(0, 2, 1)


def roll_featuremap(feature):
    assert len(feature.shape) == 3
    size = int(math.sqrt(feature.size(1)))
    return feature.permute(0, 2, 1).reshape(feature.size(0), feature.size(2), size, size)


class Downsample(nn.Module):

    def __init__(self, input_dim, hidden_dim, hw_rate, stride):
        super().__init__()

        self.maxpool = nn.MaxPool2d(hw_rate, stride=stride)
        self.avepool = nn.AvgPool2d(hw_rate, stride=stride)
        self.conv = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1),
                                  nn.BatchNorm2d(hidden_dim))

    def forward(self, x):
        return self.conv(self.maxpool(x) + self.avepool(x))


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
    

class sherl_for_resnet(nn.Module):
    def __init__(self, num_layer=4, 
                       embed_size=2048,
                       hidden_size=None,
                       hw_rate=None,
                       hw_size=None,
                       downsample_factor=2):
        super().__init__()

        self.num_layer = num_layer
        hidden_dim = embed_size // downsample_factor
        self.guide_block = Guidance(hidden_dim, hidden_dim)

        self.hor_block = nn.ModuleList(
            [Downsample(hidden_size[i], hidden_dim, hw_rate[i], hw_rate[i]) for i in range(num_layer)])

        self.gate_T = 0.1
        self.gate_params = nn.Parameter(torch.ones(1) * 0)

        self.conv_out = nn.Sequential(nn.Conv2d(hidden_dim, hidden_size[-1], kernel_size=1, stride=1),
                                      nn.BatchNorm2d(hidden_size[-1]),
                                      nn.ReLU())

    def forward(self, all_features_list, last_feature):

        assert self.num_layer == len(all_features_list)
        
        new_features_list = []
        for i, features in enumerate(all_features_list):
            new_features = flat_featuremap(self.hor_block[i](features))
            new_features_list.append(new_features)

        bs, n_instance, dim = new_features_list[0].shape

        shallow_xs = torch.stack(new_features_list[:-1], dim=2).reshape(bs * n_instance, self.num_layer - 1, dim)
        deep_x = new_features_list[-1].reshape(bs * n_instance, 1, dim)

        guide_x = self.guide_block(deep_x, shallow_xs)
        guide_x = guide_x.squeeze(1).reshape(bs, n_instance, -1)
        guide_x = roll_featuremap(guide_x)

        gate = torch.tanh(self.gate_params / self.gate_T)
        out_x = self.conv_out(guide_x) + gate * last_feature

        return out_x