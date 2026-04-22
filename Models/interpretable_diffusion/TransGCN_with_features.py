import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp


# Assuming model_utils, LearnablePositionalEncoding, Conv_MLP, AdaLayerNorm, Transpose, GELU2, series_decomp are defined elsewhere

def adjust_adj_matrix(spatial_adj, num_nodes):
    batch_size, current_size, _ = spatial_adj.shape
    if current_size == num_nodes:
        return spatial_adj  # 如果邻接矩阵已经匹配节点数，则直接返回

    padding_size = num_nodes - current_size
    identity_padding = torch.eye(padding_size, device=spatial_adj.device).expand(batch_size, -1, -1)
    spatial_adj = torch.cat([spatial_adj, torch.zeros(batch_size, current_size, padding_size, device=spatial_adj.device)], dim=2)
    spatial_adj = torch.cat([spatial_adj, torch.cat([torch.zeros(batch_size, padding_size, current_size, device=spatial_adj.device), identity_padding], dim=2)], dim=1)

    return spatial_adj

class DynamicContrastiveLearning(nn.Module):
    def __init__(self, temperature=0.5):
        super(DynamicContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features):
        # Normalize features
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eye(batch_size, device=features.device).bool()
        similarity_matrix.masked_fill_(mask, -float('inf'))
        labels = torch.arange(batch_size, device=features.device)
        loss = self.criterion(similarity_matrix, labels)
        return loss


class TrendBlock(nn.Module):
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals


class MovingBlock(nn.Module):
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float), 't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class SeasonBlock(nn.Module):
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 else (
        season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self, n_embd, condition_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        #print(x.size(), encoder_output.size())
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)
        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))  # only one really use encoder_output
        return x, att

class Encoder(nn.Module):
    def __init__(
            self,
            n_layer=14,
            n_embd=1024,
            n_head=16,
            attn_pdrop=0.,
            resid_pdrop=0.,
            mlp_hidden_times=4,
            block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season


class Decoder(nn.Module):
    def __init__(
            self,
            n_channel,
            n_feat,
            n_embd=1024,
            n_head=16,
            n_layer=10,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
            condition_dim=512
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(*[DecoderBlock(
            n_feat=n_feat,
            n_channel=n_channel,
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            condition_dim=condition_dim,
        ) for _ in range(n_layer)])

    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class AdjCombination(nn.Module):
    def __init__(self):
        super(AdjCombination, self).__init__()
        # 定义两个线性层，分别处理 adj 和 adj_new
        self.fc_adj = nn.Linear(282, 282)  # 处理原始的 adj
        self.fc_adj_new = nn.Linear(282, 282)  # 处理 adj_new

    def forward(self, adj, adj_new):
        # 扩展 adj_new 以匹配 adj 的形状
        adj_new_expanded = adj_new.expand_as(adj)

        # 通过线性层分别对 adj 和 adj_new 进行变换
        adj_transformed = torch.relu(self.fc_adj(adj))  # 对 adj 进行线性变换
        adj_new_transformed = torch.relu(self.fc_adj_new(adj_new_expanded))  # 对 adj_new 进行线性变换

        # 对两个变换后的矩阵进行组合 (例如相加或其他方式)
        combined = adj_transformed + adj_new_transformed  # 这里可以改成其他操作，如点乘等

        return combined  # 最终输出仍为 282x282


import torch
import torch.nn as nn

class CombineModule(nn.Module):
    def __init__(self, input_dim1, input_dim2, combined_dim):
        super(CombineModule, self).__init__()
        self.fc_x = nn.Linear(input_dim1, combined_dim)   # input_dim1 = 195
        self.fc_x1 = nn.Linear(input_dim2, combined_dim)  # input_dim2 = 282
        self.fc_combine = nn.Linear(combined_dim * 2, combined_dim)
        self.fc_output = nn.Linear(combined_dim, combined_dim)

    def forward(self, x, x1):
        x_transformed = torch.relu(self.fc_x(x))      # [batch_size, timesteps, combined_dim]
        x1_transformed = torch.relu(self.fc_x1(x1))   # [batch_size, timesteps, combined_dim]
        x_transformed = x_transformed.unsqueeze(2).expand(-1, -1, x1_transformed.size(2), -1)
        x1_transformed = x1_transformed.unsqueeze(-1).expand(-1, -1, -1, x_transformed.size(-1))
        combined = torch.cat((x_transformed, x1_transformed), dim=-1)  # [batch_size, timesteps, station_num, combined_dim * 2]
        x_new = torch.relu(self.fc_combine(combined))
        final_output = torch.mean(self.fc_output(x_new), dim=-1)  # [batch_size, timesteps, station_num]
        return final_output



# Adding the new adjacency matrix combination before the GCN forward pass
class GraphConvolution(nn.Module):
    def __init__(self, spatial_in_features=3, out_features=1, num_spatial_nodes=282, bias=True):
        super(GraphConvolution, self).__init__()
        self.spatial_in_features = spatial_in_features
        self.out_features = out_features
        self.num_spatial_nodes = num_spatial_nodes

        # Define learnable weights for GCN
        self.spatial_weight = nn.Parameter(torch.FloatTensor(1, num_spatial_nodes, spatial_in_features, out_features))

        # Adding the adjacency matrix combination module
        self.adj_combiner = AdjCombination()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_spatial_nodes, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.spatial_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, spatial_input, spatial_adj, adj_new):
        device = spatial_input.device

        batch_size = spatial_input.size(0)
        num_time_steps = spatial_input.size(1)
        num_spatial_nodes = self.num_spatial_nodes

        # 调整邻接矩阵大小
        new_adj = self.adj_combiner(spatial_adj, adj_new)
        spatial_input = spatial_input.to(device)
        spatial_adj = new_adj.to(device)

        # 处理每个时间步
        outputs = []
        for t in range(num_time_steps):
            spatial_input_t = spatial_input[:, t, :]  # [batch_size, num_spatial_nodes, 3]

            # Ensure spatial_input_t has the correct shape
            assert spatial_input_t.size(1) == num_spatial_nodes, f"Expected {num_spatial_nodes} nodes, got {spatial_input_t.size(1)}"

            # Expand weights to match batch size
            spatial_weight_expanded = self.spatial_weight.expand(batch_size, -1, -1, -1).to(device)

            # Reshape weights for matrix multiplication
            spatial_weight_reshaped = spatial_weight_expanded.reshape(batch_size * num_spatial_nodes, self.spatial_in_features, self.out_features)

            # Reshape input to match reshaped weight
            spatial_input_reshaped = spatial_input_t.reshape(batch_size * num_spatial_nodes, self.spatial_in_features)

            # Perform spatial convolution
            spatial_support = torch.bmm(spatial_input_reshaped.unsqueeze(1), spatial_weight_reshaped)
            spatial_support = spatial_support.squeeze(1)

            # Reshape output back to original shape
            spatial_support = spatial_support.reshape(batch_size, num_spatial_nodes, self.out_features)

            # Apply spatial adjacency matrix
            spatial_output = torch.bmm(spatial_adj, spatial_support)

            if self.bias is not None:
                spatial_output = spatial_output + self.bias.to(device)

            outputs.append(spatial_output.unsqueeze(1))  # Add the time dimension back

        final_output = torch.cat(outputs, dim=1)
        return final_output

class TransformerWithGCN(nn.Module):
    def __init__(
            self,
            n_feat1,
            n_feat,
            n_channel,
            n_layer_enc=5,
            n_layer_dec=14,
            n_embd=1024,
            n_heads=16,
            attn_pdrop=0.1,
            combined_dim=282,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
            max_len=2048,
            conv_params=None,
            num_spatial_nodes=282,
            spatial_in_features=1,
            out_features=1,  # 调整为匹配输出特征
            batch_size=32,
            **kwargs
    ):
        super().__init__()
        self.emb1 = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.emb = Conv_MLP(n_feat1, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat1, resid_pdrop=resid_pdrop)
        self.combine_module = CombineModule(input_dim1=195, input_dim2=num_spatial_nodes, combined_dim=combined_dim)
        self.emb1_x1 = nn.Linear(48, 282)

        if conv_params is None or conv_params[0] is None:
            kernel_size, padding = (1, 0) if n_feat < 32 and n_channel < 64 else (5, 2)
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0, padding_mode='circular', bias=False)
        self.res_fc = nn.Linear(144, 282)

        # Transformer 部分
        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        # GCN 部分
        self.gcn = GraphConvolution(spatial_in_features, out_features, num_spatial_nodes)
        self.gcn_fc = nn.Linear(num_spatial_nodes * out_features, 512)

        # Fully Connected Layer to combine Transformer and GCN outputs
        self.gcn_fc_feat_adjust = nn.Linear(512, 192)
        self.gcn_fc_time_adjust = nn.Linear(144, 48)
        self.fc_combined = nn.Linear(n_embd * 2, 192)  # n_embd + gcn_output 的组合

        # Contrastive learning module
        self.contrastive_learning = DynamicContrastiveLearning()

    def forward(self, x, x1, adj, adj_new, t, padding_masks=None, return_res=False):
        device = x1.device
        # x1 通过 Transformer 部分
        x1 = self.combine_module(x, x1)
        emb_x1 = self.emb1(x1)  # 仅处理 x1
        inp_enc_x1 = self.pos_enc(emb_x1)
        enc_cond_x1 = self.encoder(inp_enc_x1, t, padding_masks=padding_masks)

        inp_dec_x1 = self.pos_dec(emb_x1)
        output_x1, mean_x1, trend_x1, season_x1 = self.decoder(inp_dec_x1, t, enc_cond_x1, padding_masks=padding_masks)

        season_x1_adjusted = season_x1

        # x 通过 GCN 部分
        x_compressed = x1

        # 通过 GCN 处理 x
        gcn_output = self.gcn(spatial_adj=adj, adj_new=adj_new, spatial_input=x_compressed).to(device)

        # Adjust the shape for gcn_fc
        batch_size, time_nodes, num_spatial_nodes, out_features = gcn_output.shape
        gcn_output = gcn_output.view(batch_size * time_nodes, num_spatial_nodes * out_features)
        gcn_output = self.gcn_fc(gcn_output)
        gcn_output = gcn_output.view(batch_size, time_nodes, -1)
        if gcn_output.size(1) > x1.size(1):
            gcn_output_adjusted = self.gcn_fc_time_adjust(gcn_output.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            gcn_output_adjusted = gcn_output
        gcn_output_adjusted = self.gcn_fc_feat_adjust(gcn_output_adjusted)

        combined_output = torch.cat((output_x1, gcn_output_adjusted), dim=-1)
        combined_output = self.fc_combined(combined_output)

        res = self.inverse(combined_output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season_x1_adjusted.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(mean_x1) + res_m + trend_x1

        # Apply contrastive learning
        pooled_features = enc_cond_x1.mean(dim=1)  # 将序列特征均值池化为单一特征表示
        contrastive_loss = self.contrastive_learning(pooled_features)

        if return_res:
            return trend, self.combine_s(season_x1.transpose(1, 2)).transpose(1, 2), res - res_m, contrastive_loss

        return trend, season_error, contrastive_loss

if __name__ == '__main__':
    pass
