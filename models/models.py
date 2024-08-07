import torch
import torch.nn as nn
from models.tresnet.tresnet import TResNet
# from models.utils.registry import register_model
from .layers.avg_pool import FastAvgPool2d, FastAdaptiveAvgPool2d
from models.aggregate.layers.frame_pooling_layer import Aggregate
from models.aggregate.layers.transformer_aggregate import TAggregate
from models.aggregate.layers.transformer_agg import TAtentionAggregate
import timm
from models.attention.EMSA import EMSA
from models.attention.ExternalAttention import ExternalAttention
from models.attention.AFT import AFT_FULL
from dataset import CUFED_VIT, CUFED_VIT_CLIP

# from src.models.resnet.resnet import Bottleneck as ResnetBottleneck
# from models.resnet.resnet import ResNet

# __all__ = ['MTResnetAggregate']


class fTResNet(nn.Module):
    def __init__(self, encoder_name=None, num_classes=23, aggregate=None, args=None):
        super(fTResNet, self).__init__()
        self.encoder_name = encoder_name
        if encoder_name is not None:
            self.feature_extraction = timm.create_model(model_name=encoder_name, pretrained=True, num_classes=0).eval()
            num_feats = self.feature_extraction.num_features
        elif args.use_clip:
            num_feats = CUFED_VIT_CLIP.NUM_FEATS
        else:
            num_feats = CUFED_VIT.NUM_FEATS

        self.head = nn.Linear(num_feats, num_classes)

        # self.global_pool = FastAdaptiveAvgPool2d(flatten=True)

        if args.use_transformer:
            if args.attention == 'aft':
                attention = AFT_FULL(
                    d_model=self.feature_extraction.num_features, n=args.album_clip_length + 1)
                aggregate = TAtentionAggregate(
                    args.album_clip_length, enc_layer=attention, embed_dim=self.feature_extraction.num_features, args=args)
            elif args.attention == 'externalattention':
                attention = ExternalAttention(
                    d_model=self.feature_extraction.num_features, S=8)
                aggregate = TAtentionAggregate(
                    args.album_clip_length, enc_layer=attention, embed_dim=self.feature_extraction.num_features, args=args)
            else:
                aggregate = TAggregate(args.album_clip_length, embed_dim=num_feats, args=args)
        else:
            aggregate = Aggregate(args.album_clip_length, args=args)

        self.aggregate = aggregate

    def forward(self, x, filenames=None):
        if self.encoder_name is not None:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            with torch.no_grad():
                x = self.feature_extraction(x)
        else:
            B, N, F = x.shape
            x = x.view(B * N, F)

        # x = self.body(x)
        # self.embeddings = self.global_pool(x)

        self.embeddings = x

        if self.aggregate:
            if isinstance(self.aggregate, TAggregate):
                self.embeddings, self.attention = self.aggregate(
                    self.embeddings, filenames)
                logits = self.head(self.embeddings)
            elif isinstance(self.aggregate, TAtentionAggregate):
                self.embeddings = self.aggregate(
                    self.embeddings)
                logits = self.head(self.embeddings)
            else:  # CNN aggregation:
                logits = self.head(self.embeddings)
                logits = self.aggregate(nn.functional.softmax(logits, dim=1))

        # if attn_mat is None:
        #     return logits
        # else:
        #     return (logits, attn_mat)

        return logits, self.attention


# class fResNet(ResNet):
#     def __init__(self, aggregate=None, **kwargs):
#         super().__init__(**kwargs)
#         self.aggregate = aggregate

#     def forward(self, x):
#         x = self.body(x)
#         if self.aggregate:
#             x = self.head.global_pool(x)
#             x, attn_weight = self.aggregate(x)
#             logits = self.head.fc(self.head.FlattenDropout(x))

#         else:
#             logits = self.head(x)
#         return logits


def MTResnetAggregate(args, num_classes=23):
    aggregate = None

    model = fTResNet(encoder_name=args.backbone,
                     num_classes=num_classes, aggregate=aggregate, args=args)

    return model