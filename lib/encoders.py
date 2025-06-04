from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import math
from transformers import BertModel

from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP
import logging

import torch.nn.functional as F


logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)

def preprocess_text(text, tokenizer, max_length=64):
    inputs = tokenizer(
        text,
        truncation=True,          # 自动截断到 max_length
        padding="max_length",     # 自动填充到 max_length
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs

def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(data_name, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False, opt = None):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm, opt)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, opt=None):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        self.opt = opt

        self.attn_pool = nn.Sequential(
            nn.Linear(embed_size,1),
            nn.Softmax(dim=1))

        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.v_sa = SGE(embed_dim=embed_size, dropout_rate=0.4, num_region=36, opt=opt)
        self.gpool = GPO(32, 32)
        max_length = 36

        self.transformer_layers = nn.ModuleList([
            TransformerLayerWithRPE(embed_size, heads=8, max_length=max_length,dropout=0.1) for _ in range(4)
        ])

        self.init_weights()

    def init_weights(self):
        """Xavier 初始化全连接层的权重"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images_rg, image_lengths):
        bsize = images_rg.size(0)

        images = self.fc(images_rg)
        for layer in self.transformer_layers:
            features = layer(images, images, images)

        attn_weights = self.attn_pool(features)
        global_features = (features * attn_weights).sum(dim=1)
        rg_sa_g = self.v_sa(images_rg, global_features)
        images = rg_sa_g
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            features = self.mlp(images) + features
        features, pool_weights = self.gpool(features, image_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features



class RelativePositionEncoding(nn.Module):
    def __init__(self, max_length, embed_size):
        """
        max_length: 序列最大长度
        head_dim: 每个头的嵌入维度（需为偶数）
        """
        super().__init__()
        self.max_length = max_length
        self.head_dim = embed_size

        # 正弦函数初始化相对位置编码
        position_enc = torch.zeros(2 * max_length - 1, embed_size)
        position = torch.arange(0, 2 * max_length - 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)

        # 可学习的投影矩阵
        self.register_buffer("position_enc", position_enc)  # 不参与学习
        self.pos_proj = nn.Linear(embed_size, embed_size)

    def forward(self, length):
        """
        生成相对位置编码矩阵
        Returns: (length, length, head_dim)
        """
        length = min(length, self.max_length)

        # 生成相对位置索引矩阵 (length, length)
        range_vec = torch.arange(length,device=self.position_enc.device)
        relative_pos_matrix = range_vec[:, None] - range_vec[None, :] + (self.max_length - 1)

        # 3. 双重索引约束
        relative_pos_matrix = torch.clamp(relative_pos_matrix, min=0, max=2*self.max_length-2)
        relative_pos_matrix = relative_pos_matrix.long()  # 确保索引为整数


        # 提取编码并投影
        pos_emb = self.position_enc[relative_pos_matrix]  # (length, length, head_dim)
        return self.pos_proj(pos_emb)

class MultiHeadSelfAttentionWithRPE(nn.Module):
    def __init__(self, embed_size, heads, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim % 2 == 0, "head_dim 必须为偶数以适应正弦编码"

        # 线性投影层
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)

        # 相对位置编码
        self.relative_pos_enc = RelativePositionEncoding(max_length, embed_size)

        # Transformer-XL 全局偏置参数
        self.r_w_bias = nn.Parameter(torch.randn(heads, self.head_dim))  # 内容偏置
        self.r_r_bias = nn.Parameter(torch.randn(heads, self.head_dim))  # 位置偏置

    def rel_shift(self, x):
        """调整相对位置矩阵的偏移（避免未来信息泄漏）"""
        x_padded = torch.nn.functional.pad(x, (0, 0, 1, 0))  # 左侧填充一列
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(2) + 1, x.size(3))
        return x_padded[:, :, 1:, :]  # 形状: (N, h, q_len, k_len)

    def forward(self, values, keys, query, mask=None):
        N, q_len, _ = query.shape
        k_len = keys.size(1)

        # 投影到多头空间
        q = self.q_proj(query).view(N, q_len, self.heads, self.head_dim)
        k = self.k_proj(keys).view(N, k_len, self.heads, self.head_dim)
        v = self.v_proj(values).view(N, k_len, self.heads, self.head_dim)

        # 生成相对位置编码 (q_len, k_len, head_dim)
        rel_pos = self.relative_pos_enc(q_len)
        rel_pos = rel_pos.view(q_len, k_len, self.heads, self.head_dim).permute(2, 0, 1, 3)  # (h, q_len, k_len, d)

        # 内容项 (AC)
        q_content = q + self.r_w_bias.unsqueeze(0).unsqueeze(1)  # (N, q_len, h, d)
        ac = torch.einsum("nqhd,nkhd->nhqk", q_content, k)

        # 位置项 (BD)
        q_pos = q + self.r_r_bias.unsqueeze(0).unsqueeze(1)      # (N, q_len, h, d)
        bd = torch.einsum("nqhd,hqkd->nhqk", q_pos, rel_pos)     # (h, q_len, k_len)
        bd = self.rel_shift(bd)  # 直接处理4D张量
        # 合并得分
        attn_scores = (ac + bd) / math.sqrt(self.head_dim)

        # 应用 Mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax 和聚合
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("nhqk,nkhd->nqhd", attn_weights, v)
        out = out.reshape(N, q_len, -1)

        return self.out_proj(out)

class TransformerLayerWithRPE(nn.Module):
    def __init__(self, embed_size, heads, max_length, dropout=0.1, forward_expansion=4):
        super().__init__()
        self.attention = MultiHeadSelfAttentionWithRPE(embed_size, heads, max_length)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query ,mask=None):
        # 自注意力
        attn_out = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attn_out + query))

        # 前馈网络
        ffn_out = self.ffn(x)
        out = self.dropout(self.norm2(ffn_out + x))
        return out

#### 4-layer

class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768 * 4, embed_size)  # 扩展线性层的输入尺寸
        self.gpool = GPO(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions."""
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        outputs = self.bert(x, attention_mask=bert_attention_mask, output_hidden_states=True)

        # 使用BERT的最后四层输出进行拼接
        hidden_states = outputs.hidden_states
        last_four_layers = hidden_states[-4:]
        concatenated_output = torch.cat(last_four_layers, dim=-1)  # B x N x (4*768)

        cap_emb = self.linear(concatenated_output)  # 映射到目标维度

        pooled_features, pool_weights = self.gpool(cap_emb, lengths.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features

class SGE(nn.Module):
    def __init__(self, embed_dim, dropout_rate, num_region, opt = None):
        super(SGE, self).__init__()
        self.num_region = num_region
        self.embedding_local = nn.Sequential(nn.Linear(2048, embed_dim),
                                             nn.BatchNorm1d(self.num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

        self.opt = opt

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        new_global = self.region_attention(local, new_global)

        return new_global

    def region_attention(self, images, clip_emb):
        features_t = torch.transpose(images, 1, 2).contiguous()
        attn = torch.matmul(clip_emb.unsqueeze(1), features_t)
        attn_softmax = F.softmax(attn*self.opt.attention_lamda, dim=2)
        attn_softmax = l2norm(attn_softmax, -1)
        features = images + attn_softmax.permute(0,2,1)*(clip_emb.unsqueeze(1))

        return features

