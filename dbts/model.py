"""DBTS spatiotemporal denoiser with symmetric normalized GCN and Dynamic Contrastive Learning."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers.transformer import Transformer


def normalize_graph(a, self_loops=False):
    """Compute the symmetric degree-normalized adjacency matrix D^{-1/2} A D^{-1/2} (Eq. 9)."""
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not torch.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected a finite nonnegative square adjacency matrix.")
    if not torch.allclose(a, a.T):
        raise ValueError("Undirected station adjacency matrix required.")
    a = a.clone()
    if self_loops:
        a = a + torch.eye(len(a), device=a.device, dtype=a.dtype)
    degree = a.sum(-1)
    inv = torch.where(degree > 0, degree.clamp_min(1e-12).rsqrt(), torch.zeros_like(degree))
    return inv[:, None] * a * inv[None, :]


class GCN(nn.Module):
    """Symmetric degree-normalized graph convolution layer over metro stations (Eq. 9)."""

    def __init__(self, adj, features, output, self_loops=False):
        super().__init__()
        self.register_buffer("adj", normalize_graph(adj, self_loops))
        self.weight = nn.Linear(features, output, bias=False)

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] != len(self.adj):
            raise ValueError("GCN expects input tensor of shape [B, L, N, F].")
        return F.gelu(torch.einsum("nm,blmf->blnf", self.adj, self.weight(x)))


def contrastive(a, b, temperature=0.5, origins=None, window=None):
    """Temperature-scaled cosine-similarity contrastive loss across paired views (Eqs. 12-13)."""
    if a.shape != b.shape or len(a) < 2 or temperature <= 0:
        raise ValueError("DCL requires at least two paired samples and positive temperature.")
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / temperature
    valid = torch.ones_like(logits, dtype=torch.bool)
    if origins is not None and window is not None:
        valid = (origins[:, None] - origins[None, :]).abs() >= window
        valid.fill_diagonal_(True)
    if (valid.sum(-1) < 2).any():
        raise ValueError("No valid negative pairs found for DCL batch.")
    logits = logits.masked_fill(~valid, -torch.inf)
    return F.cross_entropy(logits, torch.arange(len(a), device=a.device))


class DBTS(nn.Module):
    """Dual-branch spatiotemporal diffusion denoiser with TDB, SDB, RB, and training-only DCL."""

    def __init__(
        self,
        nodes,
        length,
        adj,
        width=32,
        heads=4,
        enc_layers=1,
        dec_layers=1,
        dynamic_features=0,
        static_features=0,
        gcn=True,
        tdb=True,
        rb=True,
        placement="both",
        self_loops=False,
        dcl=True,
        mask_ratio=0.2,
        temperature=0.5,
        graph_residual_scale=None,
        gate_modules_by_external_features=False,
        decoder_fusion="paper_add",
    ):
        super().__init__()
        if placement not in ("encoder", "decoder", "both"):
            raise ValueError("placement must be 'encoder', 'decoder', or 'both'.")
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must lie in (0, 1).")
        if not isinstance(gate_modules_by_external_features, bool):
            raise ValueError("gate_modules_by_external_features must be a boolean.")
        if graph_residual_scale is not None and not 0 <= graph_residual_scale <= 1:
            raise ValueError("graph_residual_scale must lie in [0, 1].")
        if decoder_fusion not in ("paper_add", "concat"):
            raise ValueError("decoder_fusion must be 'paper_add' or 'concat'.")

        self.nodes = nodes
        self.length = length
        self.use_gcn = gcn
        self.use_tdb = tdb
        self.use_rb = rb
        self.placement = placement
        self.use_dcl = dcl
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        self.gate_modules_by_external_features = gate_modules_by_external_features
        self.graph_residual_scale = graph_residual_scale
        self.decoder_fusion = decoder_fusion

        self.core = Transformer(
            n_feat=nodes,
            n_channel=length,
            n_embd=width,
            n_heads=heads,
            n_layer_enc=enc_layers,
            n_layer_dec=dec_layers,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            conv_params=(1, 0),
            max_len=length,
        )
        self.dynamic = nn.Linear(dynamic_features, width, bias=False) if dynamic_features else None
        self.static = nn.Linear(nodes * static_features, width, bias=False) if static_features else None

        if gcn:
            self.graph = GCN(adj, 1 + static_features, 1, self_loops)
            self.graph_embed = nn.Linear(nodes, width)
            self.temporal_norm = nn.LayerNorm(width)
            self.spatial_norm = nn.LayerNorm(width)
            self.fuse = nn.Linear(2 * width, width)
            self.decode_nodes = nn.Linear(width, nodes)
        self.projector = nn.Linear(width, width) if dcl else None

    def module_activation(self):
        """Return active states of TDB, SDB, and RB under linked or independent ablation policy."""
        linked = self.gate_modules_by_external_features
        return {
            "policy": "external_feature_linked" if linked else "independent",
            "temporal_external_features": self.dynamic_features,
            "spatial_external_features": self.static_features,
            "tdb": bool(self.use_tdb and (not linked or self.dynamic_features > 0)),
            "sdb": bool(self.use_gcn and (not linked or self.static_features > 0)),
            "rb": bool(self.use_rb),
        }

    def conditions(self, x, dynamic, static):
        """Fuse metro flow embeddings with dynamic (weather, timetable) and static (POI, bus) inputs."""
        if self.dynamic_features:
            if dynamic is None or dynamic.shape != (x.shape[0], self.length, self.dynamic_features):
                raise ValueError("Dynamic condition shape mismatch.")
            if not torch.isfinite(dynamic).all():
                raise ValueError("Dynamic condition contains nonfinite values.")
        elif dynamic is not None:
            raise ValueError("Unexpected dynamic condition provided.")

        if self.static_features:
            if static is None or static.shape != (self.nodes, self.static_features):
                raise ValueError("Static feature shape mismatch.")
            if not torch.isfinite(static).all():
                raise ValueError("Static features contain nonfinite values.")
        elif static is not None:
            raise ValueError("Unexpected static features provided.")

        emb = self.core.emb(x)
        if dynamic is not None:
            emb = emb + self.dynamic(dynamic)
        if static is not None:
            emb = emb + self.static(static.flatten()).view(1, 1, -1)
        return emb

    def spatial(self, flow, static):
        z = flow.unsqueeze(-1)
        if static is not None:
            z = torch.cat([z, static[None, None].expand(*z.shape[:2], -1, -1)], dim=-1)
        return self.graph_embed(self.graph(z).squeeze(-1))

    def fuse_graph(self, temporal, flow, static):
        fused = self.fuse(torch.cat([self.temporal_norm(temporal), self.spatial_norm(self.spatial(flow, static))], -1))
        return fused if self.graph_residual_scale is None else temporal + self.graph_residual_scale * fused

    def encode(self, x, t, dynamic=None, static=None):
        """Dual-branch encoder combining Transformer temporal path and GCN spatial path (Fig. 2)."""
        emb = self.conditions(x, dynamic, static)
        active = self.module_activation()
        enc = self.core.pos_enc(emb)
        if active["tdb"]:
            enc = self.core.encoder(enc, t)
        else:
            enc = self.core.encoder.blocks[0].ln1(enc, t)
        if active["sdb"] and self.placement in ("encoder", "both"):
            enc = self.fuse_graph(enc, x, static)
        return enc, emb

    def forward(self, x, t, dynamic=None, static=None, parts=False):
        """Spatiotemporal decoder with parallel TDB+SDB fusion (Eq. 10) and decomposition (Eq. 19)."""
        if x.ndim != 3 or x.shape[1:] != (self.length, self.nodes):
            raise ValueError("Expected input tensor of shape [B, L, N].")
        enc, emb = self.encode(x, t, dynamic, static)
        active = self.module_activation()
        z = self.core.pos_dec(emb)
        means = []
        trend = torch.zeros_like(x)
        season = torch.zeros_like(z)

        for block in self.core.decoder.blocks:
            use_spatial = active["sdb"] and self.placement in ("decoder", "both")
            spatial = None
            if use_spatial and self.decoder_fusion == "paper_add":
                spatial = self.spatial(self.decode_nodes(z), static)
            if active["tdb"]:
                a, _ = block.attn1(block.ln1(z, t))
                z = z + a
                a, _ = block.attn2(block.ln1_1(z, t), enc)
                z = z + a
            else:
                z = block.ln1(z, t) + enc
            if spatial is not None:
                z = z + spatial
            elif use_spatial:
                z = self.fuse_graph(z, self.decode_nodes(z), static)
            if self.use_rb:
                z1, z2 = block.proj(z).chunk(2, dim=1)
                trend = trend + block.trend(z1)
                season = season + block.seasonal(z2)
            z = z + block.mlp(block.ln2(z))
            mean = z.mean(1, keepdim=True)
            z = z - mean
            means.append(block.linear(mean))

        residual = self.core.inverse(z)
        rm = residual.mean(1, keepdim=True)
        if self.use_rb:
            trend = trend + self.core.combine_m(torch.cat(means, dim=1)) + rm
            periodic = self.core.combine_s(season.transpose(1, 2)).transpose(1, 2)
            residual = residual - rm
        else:
            trend = torch.zeros_like(x)
            periodic = torch.zeros_like(x)
        prediction = trend + periodic + residual
        return (prediction, {"trend": trend, "periodic": periodic, "residual": residual}) if parts else prediction

    def dcl_loss(self, clean, dynamic=None, static=None, origins=None):
        """Training-only Dynamic Contrastive Learning on shared encoder and projector (Fig. 6, Eqs. 12-13)."""
        if not self.use_dcl:
            return clean.new_zeros(())
        keep = torch.rand_like(clean) > self.mask_ratio
        t = torch.zeros(len(clean), dtype=torch.long, device=clean.device)
        a, _ = self.encode(clean, t, dynamic, static)
        b, _ = self.encode(clean * keep, t, dynamic, static)
        return contrastive(self.projector(a.mean(1)), self.projector(b.mean(1)), self.temperature, origins, self.length)


class SequenceDiffusionBaseline(nn.Module):
    """Sequence-only Transformer diffusion baseline without spatial GCN or DCL modules."""

    def __init__(self, nodes, length, width=32, heads=4, enc_layers=1, dec_layers=1):
        super().__init__()
        self.nodes = nodes
        self.length = length
        self.use_dcl = False
        self.core = Transformer(
            n_feat=nodes,
            n_channel=length,
            n_embd=width,
            n_heads=heads,
            n_layer_enc=enc_layers,
            n_layer_dec=dec_layers,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            conv_params=(1, 0),
            max_len=length,
        )

    def forward(self, x, t, dynamic=None, static=None):
        trend, periodic = self.core(x, t)
        return trend + periodic
