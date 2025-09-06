import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree, softmax


def get_mask(input_size: int, window_size: List[int]) -> List[int]:
    """Compute the temporal length at each scale after windowed down-sampling.

    Example: input_size=96, window_size=[4,4] -> [96, 24, 6]
    """
    all_size = [input_size]
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)
    return all_size


class MultiScaleTemporalPooling(nn.Module):
    """Multi-scale temporal downsampling using AvgPool1d along the time axis.

    Input: x [B, T, D]
    Output: list of tensors per scale: [x_s0 [B, T0, D], x_s1 [B, T1, D], ...]
    where T0=T, T1=floor(T/window_size[0]), etc.
    """

    def __init__(self, window_size: List[int]):
        super().__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, T, D]
        outs = [x]
        cur = x
        for k in self.window_size:
            if k <= 1:
                outs.append(cur)
                continue
            # Pool along time axis
            cur = F.avg_pool1d(cur.transpose(1, 2), kernel_size=k, stride=k, ceil_mode=False).transpose(1, 2)
            outs.append(cur)
        return outs


class MultiAdaptiveHypergraph(nn.Module):
    """Learned, scale-wise adaptive hypergraph builder over temporal nodes.

    For each scale i, build an incidence by node/hyper-edge embeddings:
    - node embeddings size = temporal length at scale i
    - hyper-edge embeddings count = hyper_num[i]
    - softmax(ReLU(alpha * node@edge^T)), keep top-k per node, threshold, and form indices.
    Returns: list of hyperedge_index tensors [2, E_i] on the given device.
    """

    def __init__(self, seq_len: int, window_size: List[int], d_model: int, hyper_num: List[int], k: int = 10, alpha: float = 3.0):
        super().__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.d_model = d_model
        self.hyper_num = hyper_num
        self.k = k
        self.alpha = alpha

        # Precompute temporal lengths per scale
        self.all_size = get_mask(seq_len, window_size)

        self.embed_edge = nn.ModuleList()
        self.embed_node = nn.ModuleList()
        for i in range(len(self.hyper_num)):
            self.embed_edge.append(nn.Embedding(self.hyper_num[i], d_model))
            self.embed_node.append(nn.Embedding(self.all_size[i], d_model))

    @torch.no_grad()
    def build(self, device: torch.device) -> List[torch.Tensor]:
        hyperedge_all: List[torch.Tensor] = []
        for i in range(len(self.hyper_num)):
            num_nodes = self.all_size[i]
            num_edges = self.hyper_num[i]

            node_idx = torch.arange(num_nodes, device=device)
            edge_idx = torch.arange(num_edges, device=device)

            node_ec = self.embed_node[i](node_idx)  # [T_i, D]
            edge_ec = self.embed_edge[i](edge_idx)  # [E_i, D]

            # Affinity [T_i, E_i]
            a = node_ec @ edge_ec.transpose(0, 1)
            adj = F.softmax(F.relu(self.alpha * a), dim=1)

            # Keep top-k per node
            k = min(self.k, adj.size(1))
            topv, tope = torch.topk(adj, k, dim=1)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, tope, 1.0)
            adj = adj * mask

            # Threshold to binary and remove empty edges
            adj_bin = (adj > 0.5).to(torch.int64)
            keep_edges = torch.any(adj_bin != 0, dim=0)
            if keep_edges.sum() == 0:
                # Fallback: connect each node to its own edge (identity-like)
                node_list = torch.arange(num_nodes, device=device, dtype=torch.long)
                edge_list = torch.arange(num_nodes, device=device, dtype=torch.long)
                hyperedge_index = torch.stack([node_list, edge_list], dim=0)
                hyperedge_all.append(hyperedge_index)
                continue

            adj_bin = adj_bin[:, keep_edges]
            # Build COO indices: for each edge j, pick nodes where adj_bin[:, j]==1
            node_list = []
            edge_list = []
            for j in range(adj_bin.size(1)):
                nodes = torch.nonzero(adj_bin[:, j], as_tuple=False).flatten()
                if nodes.numel() == 0:
                    continue
                node_list.append(nodes)
                edge_list.append(torch.full((nodes.numel(),), j, device=device, dtype=torch.long))

            if len(node_list) == 0:
                # Fallback again in rare case after filtering
                node_list = torch.arange(num_nodes, device=device, dtype=torch.long)
                edge_list = torch.arange(num_nodes, device=device, dtype=torch.long)
            else:
                node_list = torch.cat(node_list, dim=0)
                edge_list = torch.cat(edge_list, dim=0)

            hyperedge_index = torch.stack([node_list, edge_list], dim=0)  # [2, E]
            hyperedge_all.append(hyperedge_index)

        return hyperedge_all


class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_attention: bool = True,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.1,
                 bias: bool = False):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self, x: torch.Tensor, hyperedge_index: torch.Tensor, alpha: torch.Tensor = None):
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges / 2), x.dtype)
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        return out

    def message(self, x_j: torch.Tensor, edge_index_i: torch.Tensor, norm: torch.Tensor, alpha: torch.Tensor):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N_t, C]
        x = torch.matmul(x, self.weight)  # [B, N_t, C]
        x1 = x.transpose(0, 1)  # [N_t, B, C]

        # Node and hyperedge aggregation pre-attention
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])  # [E, B, C]

        # Build per-edge sum of node features for attention pairing
        edge_sums = {}
        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            e = edge_id.item()
            n = node_id.item()
            if e not in edge_sums:
                edge_sums[e] = x1[n, :, :]
            else:
                edge_sums[e] += x1[n, :, :]

        if len(edge_sums) == 0:
            # Degenerate fallback
            out = x
            constrain_losstotal = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return out, constrain_losstotal

        result_list = torch.stack([value for value in edge_sums.values()], dim=0)  # [E_u, B, C]
        # Map hyperedge_index[1] into continuous 0..E_u-1 indexing
        unique_edges = torch.tensor(list(edge_sums.keys()), device=x.device, dtype=torch.long)
        remap = {int(u.item()): i for i, u in enumerate(unique_edges)}
        mapped_edge_idx = torch.tensor([remap[int(e.item())] for e in hyperedge_index[1]], device=x.device, dtype=torch.long)
        x_j = torch.index_select(result_list, dim=0, index=mapped_edge_idx)  # [E, B, C]

        # Hyperedge consistency loss (pairwise)
        loss_hyper = 0.0
        keys = list(edge_sums.keys())
        for k in range(len(keys)):
            for m in range(len(keys)):
                ek = edge_sums[keys[k]]
                em = edge_sums[keys[m]]
                inner_product = torch.sum(ek * em, dim=1, keepdim=True)
                norm_q_i = torch.norm(ek, dim=1, keepdim=True) + 1e-6
                norm_q_j = torch.norm(em, dim=1, keepdim=True) + 1e-6
                alpha = inner_product / (norm_q_i * norm_q_j)
                distan = torch.norm(ek - em, dim=1, keepdim=True)
                loss_item = alpha * distan + (1 - alpha) * torch.clamp(torch.tensor(4.2, device=x.device, dtype=x.dtype) - distan, min=0.0)
                loss_hyper += torch.abs(torch.mean(loss_item))

        loss_hyper = loss_hyper / ((len(edge_sums) + 1) ** 2)

        # Attention coefficients
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [E, B, heads]
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))
        alpha = F.dropout(alpha, p=0.1, training=self.training)

        D = degree(hyperedge_index[0], x1.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        Bdeg = 1.0 / degree(hyperedge_index[1], int(num_edges / 2), x.dtype)
        Bdeg[Bdeg == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=Bdeg, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out = out.transpose(0, 1)  # [B, N_t, C]

        constrain_loss = x_i - x_j
        constrain_lossfin1 = torch.mean(constrain_loss)
        constrain_losstotal = torch.abs(constrain_lossfin1) + loss_hyper
        return out, constrain_losstotal

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_weight = nn.Linear(dim, dim)
        self.key_weight = nn.Linear(dim, dim)
        self.value_weight = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, E, D]
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5 + 1e-6), dim=-1)
        attended_values = torch.matmul(attention_scores, v)
        return attended_values


class MultiScaleHyperTemporalHead(nn.Module):
    """Residual temporal forecaster using multi-scale adaptive hypergraphs.

    Produces a node-agnostic temporal pattern forecast [B, H, output_dim] and
    broadcasts it over nodes, to be fused with a node-wise spatiotemporal model.
    """

    def __init__(self,
                 seq_len: int,
                 horizon: int,
                 input_dim: int,
                 output_dim: int,
                 window_size: List[int],
                 hyper_num: List[int],
                 k: int = 10,
                 alpha: float = 3.0):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size

        self.all_size = get_mask(seq_len, window_size)
        self.total_nodes = sum(self.all_size)

        self.pool = MultiScaleTemporalPooling(window_size)
        self.builder = MultiAdaptiveHypergraph(seq_len, window_size, d_model=input_dim, hyper_num=hyper_num, k=k, alpha=alpha)
        self.hyconv = nn.ModuleList([HypergraphConv(input_dim, input_dim) for _ in range(len(hyper_num))])

        # Simple temporal projections
        self.base_linear = nn.Linear(seq_len, horizon)
        self.out_tran = nn.Linear(self.total_nodes, horizon)
        self.refine = nn.Linear(horizon, horizon)
        self.channel_tran = nn.Linear(input_dim, output_dim)
        self.attn = SelfAttentionLayer(input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        device = x.device

        # Node-agnostic temporal series
        xs = x.mean(dim=2)  # [B, T, D]

        # Normalize across time like ASHyper
        mean_enc = xs.mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(torch.var(xs, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        xs_n = (xs - mean_enc) / std_enc

        # Scales
        seq_list = self.pool(xs_n)  # list of [B, L_i, D]

        # Hypergraphs (per scale)
        hyper_list = self.builder.build(device)

        sum_hyper_list = []
        result_tensor = None
        result_conloss = None

        for i in range(len(hyper_list)):
            he = hyper_list[i].to(device)
            node_value = seq_list[i]  # [B, L_i, D]

            # Sum features per hyperedge for inter-scale attention later
            edge_sums = {}
            for edge_id, node_id in zip(he[1], he[0]):
                e = edge_id.item()
                n = node_id.item()
                if e not in edge_sums:
                    edge_sums[e] = node_value[:, n, :]
                else:
                    edge_sums[e] += node_value[:, n, :]

            for _, sum_value in edge_sums.items():
                sum_hyper_list.append(sum_value.unsqueeze(1))  # [B, 1, D]

            # Intra-scale hypergraph conv
            out_i, conloss_i = self.hyconv[i](node_value, he)  # [B, L_i, D]
            result_tensor = out_i if result_tensor is None else torch.cat([result_tensor, out_i], dim=1)
            result_conloss = conloss_i if result_conloss is None else (result_conloss + conloss_i)

        if len(sum_hyper_list) > 0:
            sum_hyper = torch.cat(sum_hyper_list, dim=1)  # [B, E_tot, D]
            attn_out = self.attn(sum_hyper)  # [B, E_tot, D]
            # Pool across hyperedges to horizon
            inter_out = F.adaptive_avg_pool1d(attn_out.transpose(1, 2), output_size=self.horizon).transpose(1, 2)  # [B, H, D]
        else:
            inter_out = torch.zeros(B, self.horizon, D, device=device, dtype=x.dtype)

        # Project concatenated intra-scale outputs to horizon
        if result_tensor is None:
            intra_out = torch.zeros(B, self.horizon, D, device=device, dtype=x.dtype)
        else:
            # [B, sum(L_i), D] -> [B, D, sum(L_i)] -> Linear -> [B, D, H] -> [B, H, D]
            intra_out = self.out_tran(result_tensor.permute(0, 2, 1)).permute(0, 2, 1)

        # Base linear forecast from input
        base_out = self.base_linear(xs_n.permute(0, 2, 1)).permute(0, 2, 1)  # [B, H, D]

        y = base_out + intra_out + inter_out
        y = self.refine(y.transpose(1, 2)).transpose(1, 2)  # small refinement on H
        # De-normalize
        y = y * std_enc + mean_enc

        # Channel projection and broadcast to nodes
        y = self.channel_tran(y)  # [B, H, output_dim]
        y = y.unsqueeze(2).expand(-1, -1, N, -1)  # [B, H, N, output_dim]

        if result_conloss is None:
            result_conloss = torch.tensor(0.0, device=device, dtype=x.dtype)
        return y, result_conloss
