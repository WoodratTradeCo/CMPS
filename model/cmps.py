import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
from einops import rearrange, repeat
from model.mobilenet_v3 import mobilenetv3_large, h_swish


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: scale factor
        Return:
            self-attention
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # torch.Size([16, 149, 149])
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)

        context = torch.matmul(attention, V)  # torch.Size([16, 149, 64])

        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.batch_norm = nn.BatchNorm1d(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)  # torch.Size([2, 149, 512])
        K = self.fc_K(x)

        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        # out = self.layer_norm(out)
        out = self.batch_norm(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.1):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.batch_norm = nn.BatchNorm1d(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.batch_norm(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        return out


class PatchEmbed(nn.Module):
    """
      Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x  # x.shape is [8, 196, 768]


class CMPS(nn.Module):
    def __init__(self, cnn, embed_dim=512, num_layers=3, hidden=1024, num_classes=345, img_channel=960, num_patches=49,
                 num_strokes=100):
        super(CMPS, self).__init__()
        self.deep_stroke_embedding = nn.GRU(input_size=4, hidden_size=256, num_layers=4, batch_first=True,
                                            dropout=0.5, bidirectional=True)

        self.cnn = cnn
        self.hidden = hidden
        self.stroke_encoder = Encoder(dim_model=embed_dim, num_head=8, hidden=self.hidden, dropout=0.3)
        self.multimodal_encoder = Encoder(dim_model=embed_dim, num_head=8, hidden=self.hidden, dropout=0.2)

        self.stroke_encoders = nn.ModuleList([
            copy.deepcopy(self.stroke_encoder)
            for _ in range(num_layers)])

        self.multimodal_encoders = nn.ModuleList([
            copy.deepcopy(self.multimodal_encoder)
            for _ in range(num_layers)])

        self.img_pos_token = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)
        self.stroke_pos_token = nn.Parameter(torch.zeros(1, num_strokes, embed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.sep_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        self.conv_embed = nn.Sequential(nn.Conv1d(img_channel, embed_dim, 1))
        self.conv_embed_identity = nn.Sequential(nn.Conv2d(embed_dim, img_channel, 1))
        self.stroke_mlp_head = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.img_mlp_head = nn.Sequential(
            # nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.img_classifier = nn.Sequential(
            nn.Linear(embed_dim + img_channel, 1280),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )
        self.fc = nn.Linear(embed_dim + img_channel, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = nn.Parameter(torch.full((1, num_classes), 0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.full((1, num_classes), 0.5), requires_grad=True)

    def forward(self, stroke, img):

        b, _, _ = stroke.shape
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        # sep_tokens = repeat(self.sep_token, '1 1 d -> b 1 d', b=b)
        # stroke embedding
        deep_stroke_embed, _ = self.deep_stroke_embedding(stroke)

        stroke_pos_encoding = self.stroke_pos_token  # position encoding
        stroke_embed = deep_stroke_embed + stroke_pos_encoding  # stroke tokens

        for stroke_encoder in self.stroke_encoders:
            stroke_embed = stroke_encoder(stroke_embed)
        dse_identity = stroke_embed
        # pixel embedding
        img = self.cnn.features(img)
        img_conv = self.cnn.conv(img)

        img_embed = torch.flatten(img_conv, 2)
        img_embed = self.conv_embed(img_embed)
        img_embed = img_embed.permute(0, 2, 1)

        kl_img = img_embed.mean(dim=1)
        kl_stroke = dse_identity.mean(dim=1)

        img_embed = img_embed + self.img_pos_token
        # embeddings concat

        stroke_img = torch.cat((stroke_embed, img_embed), dim=1)
        for multimodal_encoder in self.multimodal_encoders:
            stroke_img = multimodal_encoder(stroke_img)
        #
        multimodal_stroke = stroke_img[:, :100]
        multimodal_img = stroke_img[:, 100:]
        stroke_att = self.softmax(self.avgpool(multimodal_stroke))
        output_s = stroke_att * dse_identity + dse_identity
        output_s = self.stroke_mlp_head(output_s.mean(dim=1))

        multimodal_img = multimodal_img.permute(0, 2, 1)
        multimodal_img = multimodal_img.reshape(b, multimodal_img.size(1), img_conv.size(2), -1)
        output_i = torch.cat((multimodal_img, img_conv), dim=1)
        output_i = self.cnn.avgpool(output_i)
        output_i = output_i.view(output_i.size(0), -1)
        output_i = self.img_classifier(output_i)  # torch.Size([2, 345])
        output_c = self.alpha * output_i + self.beta * output_s
        return output_i, output_s, output_c, kl_img, kl_stroke
