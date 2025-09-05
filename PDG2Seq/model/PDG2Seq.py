import torch
import torch.nn as nn
from model.PDG2SeqCell import PDG2SeqCell
import numpy as np
class PDG2Seq_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PDG2Seq_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim,
                                             use_hypergraph=getattr(self, 'use_hypergraph', True),
                                             use_interactive=getattr(self, 'use_interactive', True),
                                             num_hyper_edges=getattr(self, 'num_hyper_edges', 32)))
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim,
                                                 use_hypergraph=getattr(self, 'use_hypergraph', True),
                                                 use_interactive=getattr(self, 'use_interactive', True),
                                                 num_hyper_edges=getattr(self, 'num_hyper_edges', 32)))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]     #x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]   #state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):   #如果有两层GRU，则第二层的GGRU的输入是前一层的隐藏状态
                state = self.PDG2Seq_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :], node_embeddings[1][:, t, :], node_embeddings[2]])#state=[batch,steps,nodes,input_dim]
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state,[node_embeddings[0], node_embeddings[1]])
                inner_states.append(state)   #一个list，里面是每一步的GRU的hidden状态
            output_hidden.append(state)  #每层最后一个GRU单元的hidden状态
            current_inputs = torch.stack(inner_states, dim=1)
            #拼接成完整的上一层GRU的hidden状态，作为下一层GRRU的输入[batch,steps,nodes,hiddensize]
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.PDG2Seq_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class PDG2Seq_Dncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PDG2Seq_Dncoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim,
                                             use_hypergraph=getattr(self, 'use_hypergraph', True),
                                             use_interactive=getattr(self, 'use_interactive', True),
                                             num_hyper_edges=getattr(self, 'num_hyper_edges', 32)))
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim,
                                                 use_hypergraph=getattr(self, 'use_hypergraph', True),
                                                 use_interactive=getattr(self, 'use_interactive', True),
                                                 num_hyper_edges=getattr(self, 'num_hyper_edges', 32)))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.PDG2Seq_cells[i](current_inputs, init_state[i], [node_embeddings[0], node_embeddings[1], node_embeddings[2]])
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class PDG2Seq(nn.Module):
    def __init__(self, args):
        super(PDG2Seq, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.cl_decay_steps = args.lr_decay_step
        self.node_embeddings1 = nn.Parameter(torch.empty(self.num_node, args.embed_dim))
        self.T_i_D_emb1 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(7, args.time_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(7, args.time_dim))

        # Multi-scale pooling ratios
        self.pool_ratios = getattr(args, 'pool_ratios', [1, 3, 6])
        self.pooling_layers = nn.ModuleList([
            nn.AvgPool1d(ratio, stride=ratio) if ratio > 1 else nn.Identity()
            for ratio in self.pool_ratios
        ])

        # Fusion layers for global/local features
        fusion_dim = self.hidden_dim * len(self.pool_ratios)

        self.encoder = PDG2Seq_Encoder(
            args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers
        )
        self.encoder.use_hypergraph = getattr(args, 'use_hypergraph', True)
        self.encoder.use_interactive = getattr(args, 'use_interactive', True)
        self.encoder.num_hyper_edges = getattr(args, 'num_hyper_edges', 32)

        self.decoder = PDG2Seq_Dncoder(
            args.num_nodes, fusion_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers
        )
        self.decoder.use_hypergraph = getattr(args, 'use_hypergraph', True)
        self.decoder.use_interactive = getattr(args, 'use_interactive', True)
        self.decoder.num_hyper_edges = getattr(args, 'num_hyper_edges', 32)
        #predictor
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        # Fusion layers for global/local features
        fusion_dim = self.hidden_dim * len(self.pool_ratios)
        self.global_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU()
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU()
        )

    def forward(self, source, traget=None, batches_seen=None):
        # Multi-scale pooling and fusion
        t_i_d_data1 = source[..., 0,-2]
        t_i_d_data2 = traget[..., 0,-2]
        T_i_D_emb1_en = self.T_i_D_emb1[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb2_en = self.T_i_D_emb2[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb1_de = self.T_i_D_emb1[(t_i_d_data2 * 288).type(torch.LongTensor)]
        T_i_D_emb2_de = self.T_i_D_emb2[(t_i_d_data2 * 288).type(torch.LongTensor)]
        if self.use_W:
            d_i_w_data1 = source[..., 0,-1]
            d_i_w_data2 = traget[..., 0,-1]
            D_i_W_emb1_en = self.D_i_W_emb1[(d_i_w_data1).type(torch.LongTensor)]
            D_i_W_emb2_en = self.D_i_W_emb2[(d_i_w_data1).type(torch.LongTensor)]
            D_i_W_emb1_de = self.D_i_W_emb1[(d_i_w_data2).type(torch.LongTensor)]
            D_i_W_emb2_de = self.D_i_W_emb2[(d_i_w_data2).type(torch.LongTensor)]
            node_embedding_en1 = torch.mul(T_i_D_emb1_en, D_i_W_emb1_en)
            node_embedding_en2 = torch.mul(T_i_D_emb2_en, D_i_W_emb2_en)
            node_embedding_de1 = torch.mul(T_i_D_emb1_de, D_i_W_emb1_de)
            node_embedding_de2 = torch.mul(T_i_D_emb2_de, D_i_W_emb2_de)
        else:
            node_embedding_en1 = T_i_D_emb1_en
            node_embedding_en2 = T_i_D_emb2_en
            node_embedding_de1 = T_i_D_emb1_de
            node_embedding_de2 = T_i_D_emb2_de

        en_node_embeddings=[node_embedding_en1, node_embedding_en2, self.node_embeddings1]
        source = source[..., :self.input_dim]
        init_state = self.encoder.init_hidden(source.shape[0]).to(source.device)
        state, _ = self.encoder(source, init_state, en_node_embeddings)

        # Multi-scale pooling
        pooled_states = []
        for i, pool in enumerate(self.pooling_layers):
            # Pool over time dimension (B, T, N, hidden) -> (B, T', N, hidden)
            pooled = pool(state.permute(0,2,3,1).reshape(-1, state.shape[1])).reshape(state.shape[0], state.shape[2], state.shape[3], -1).permute(0,3,1,2)
            pooled_states.append(pooled)
        # Concatenate pooled features
        pooled_cat = torch.cat([p[:, -1, :, :] for p in pooled_states], dim=-1)  # local
        pooled_mean = torch.cat([p.mean(dim=1) for p in pooled_states], dim=-1)  # global
        local_feature = self.local_fusion(pooled_cat)
        global_feature = self.global_fusion(pooled_mean)
        fusion_feature = torch.cat([local_feature, global_feature], dim=-1)

        ht_list = [torch.zeros((source.shape[0], self.num_node, self.hidden_dim), device=source.device)] * self.num_layers
        fusion_dim = self.hidden_dim * len(self.pool_ratios)
        decoder_input = fusion_feature  # first input to decoder
        out = []
        for t in range(self.horizon):
            state, ht_list = self.decoder(decoder_input, ht_list, [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1])
            go = self.proj(state)
            out.append(go)
            # Next decoder input: use fusion_feature for t=0, then use go for subsequent steps
            if t == 0:
                decoder_input = go.new_zeros((source.shape[0], self.num_node, fusion_dim))
            else:
                decoder_input = go.new_zeros((source.shape[0], self.num_node, fusion_dim))
        output = torch.stack(out, dim=1)
        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x
