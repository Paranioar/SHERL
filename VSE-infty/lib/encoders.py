"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from transformers import BertModel

from lib.utils import l2norm, count_params, random_drop_feature
from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP
from lib.adapter_for_transformer import sherl_for_transformer

import logging
logger = logging.getLogger(__name__)


def get_text_encoder(opt, embed_size, no_txtnorm=False):
    return EncoderText(opt, embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(opt, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        img_enc = EncoderImageFull(
            opt, backbone_source, backbone_path, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.gpool = GPO(32, 32)
        self.init_weights()

        self.need_train_params = list(self.parameters())

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        features = self.mlp(images) + features

        features, pool_weights = self.gpool(features, image_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


class EncoderImageFull(nn.Module):
    def __init__(self, opt, backbone_source, backbone_path, img_dim, embed_size, precomp_enc_type='backbone', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        self.need_train_params = list()
        self.image_agg = GPO(32, 32)
        self.need_train_params.extend(list(self.image_agg.parameters()))

        self.image_fc = nn.Linear(img_dim, embed_size)
        self.xavier_initfc(self.image_fc)
        self.need_train_params.extend(list(self.image_fc.parameters()))

        self.backbone = ResnetFeatureExtractor(opt, backbone_source, backbone_path)
        
        for name, param in self.backbone.named_parameters():
            if 'side' in name:
                param.requires_grad = True
                self.need_train_params.append(param)
            else:
                param.requires_grad = False

    def xavier_initfc(self, fc_layer):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(fc_layer.in_features +
                                  fc_layer.out_features)
        fc_layer.weight.data.uniform_(-r, r)
        fc_layer.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        
        features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            features, feat_lengths = random_drop_feature(features, 0.2)
        else:
            feat_lengths = torch.zeros(features.size(0)).to(features.device)
            feat_lengths[:] = features.size(1)

        features = self.image_fc(features)
        features, _ = self.image_agg(features, feat_lengths)
        features = l2norm(features, dim=-1)

        return features


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, opt, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()

        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.need_train_params = list()
        self.text_agg = GPO(32, 32)
        self.need_train_params.extend(list(self.text_agg.parameters()))

        self.text_fc = nn.Linear(768, embed_size)
        self.need_train_params.extend(list(self.text_fc.parameters()))

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              local_files_only=True)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.side_module = sherl_for_transformer(num_layer=12,
                                                 input_dim=768,
                                                 embed_size=opt.embed_size,
                                                 downsample_factor=opt.downsample_factor)
        self.need_train_params.extend(list(self.side_module.parameters()))
        self.num_layer = 12

    def forward(self, input_ids, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        features_list = list()
        attention_mask = (input_ids != 0).float()

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        features = self.bert.embeddings(input_ids=input_ids,
                                        position_ids=None,
                                        token_type_ids=token_type_ids,
                                        inputs_embeds=None,
                                        past_key_values_length=0)

        features_list.append(features)
        for i, layer_module in enumerate(self.bert.encoder.layer):
            
            if i < self.num_layer - 1:
                hidden_states = layer_module(features, extended_attention_mask, None, None, None, None, False)
                features = hidden_states[0]
                features_list.append(features)
            else:
                features = self.side_module(features_list, features)
                hidden_states = layer_module(features, extended_attention_mask, None, None, None, None, False)
                features = hidden_states[0]

        features = self.text_fc(features)
        features, _ = self.text_agg(features, lengths.to(device))
        features = l2norm(features, dim=-1)

        return features


if __name__ == '__main__':
    import os 
    from arguments import get_argument_parser
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    parser = get_argument_parser()
    opt = parser.parse_args()
    model = EncoderText(opt, 1024, False).cuda()
    # tensorA = torch.Tensor([[i for i in range(101, 121)] for _ in range(13)]).long().cuda()
    # lens = torch.FloatTensor([10, 15, 19, 16, 9, 10, 15, 19, 16, 9, 10, 15, 19]).cuda()
    # output = model(tensorA, lens)
    print(model.bert.encoder)
    print('finished')
    