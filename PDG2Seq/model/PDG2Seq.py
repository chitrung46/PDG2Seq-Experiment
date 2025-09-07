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
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
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

        self.encoder = PDG2Seq_Encoder(
            args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers
        )
        self.encoder.use_hypergraph = getattr(args, 'use_hypergraph', True)
        self.encoder.use_interactive = getattr(args, 'use_interactive', True)
        self.encoder.num_hyper_edges = getattr(args, 'num_hyper_edges', 32)

        self.decoder = PDG2Seq_Dncoder(
            args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers
        )
        self.decoder.use_hypergraph = getattr(args, 'use_hypergraph', True)
        self.decoder.use_interactive = getattr(args, 'use_interactive', True)
        self.decoder.num_hyper_edges = getattr(args, 'num_hyper_edges', 32)
        #predictor
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        # multi-scale options
        self.use_multi_scale = getattr(args, 'use_multi_scale', False)
        # define scales (ratios) to pool the temporal dimension. default matches example: [1,2,3,4,6,12]
        self._scales = getattr(args, 'scales', [1, 2, 3, 4, 6, 12]) if self.use_multi_scale else [1]
        # build one encoder per scale (if scale>1 we will apply TemporalPooling before encoding)
        self.encoders = nn.ModuleList()
        for _ in self._scales:
            enc = PDG2Seq_Encoder(
                args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                args.embed_dim, args.time_dim, args.num_layers
            )
            enc.use_hypergraph = getattr(args, 'use_hypergraph', True)
            enc.use_interactive = getattr(args, 'use_interactive', True)
            enc.num_hyper_edges = getattr(args, 'num_hyper_edges', 32)
            self.encoders.append(enc)

        # fusion layers to combine multi-scale local/global features
        n_paths = len(self._scales)
        self.local_fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * n_paths, self.hidden_dim),
            nn.ReLU()
        )
        self.global_fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * n_paths, self.hidden_dim),
            nn.ReLU()
        )
        # project fused local+global -> decoder initial hidden dim
        self.fusion_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)


class TemporalPooling(nn.Module):
    def __init__(self, mode='mean', ratio=2):
        super(TemporalPooling, self).__init__()
        self.mode = mode
        self.ratio = ratio

    def forward(self, x):
        # x: B x T x N x D
        if self.ratio == 1:
            return x
        # trim length to multiple of ratio by padding at the front with zeros if needed
        b, t, n, d = x.shape
        rem = t % self.ratio
        if rem != 0:
            pad_len = self.ratio - rem
            pad = torch.zeros(b, pad_len, n, d, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
            t = x.size(1)
        x = x.reshape(b, -1, self.ratio, n, d)
        if self.mode == 'max':
            y = x.max(dim=2)[0]
        else:
            y = x.mean(dim=2)
        return y

    def forward(self, source, traget=None, batches_seen=None):
        #source: B, T_1, N, D
        #target: B, T_2, N, D


        t_i_d_data1 = source[..., 0,-2]
        t_i_d_data2 = traget[..., 0,-2]
        # T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        T_i_D_emb1_en = self.T_i_D_emb1[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb2_en = self.T_i_D_emb2[(t_i_d_data1 * 288).type(torch.LongTensor)]

        T_i_D_emb1_de = self.T_i_D_emb1[(t_i_d_data2 * 288).type(torch.LongTensor)]
        T_i_D_emb2_de = self.T_i_D_emb2[(t_i_d_data2 * 288).type(torch.LongTensor)]
        if self.use_W:
            d_i_w_data1 = source[..., 0,-1]
            d_i_w_data2 = traget[..., 0,-1]
            # D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
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

        init_state = self.encoder.init_hidden(source.shape[0]).to(source.device)  # [2,64,307,64] 前面是2是因为有两层GRU
        state, _ = self.encoder(source, init_state, en_node_embeddings)  # B, T, N, hidden
        state = state[:, -1:, :, :].squeeze(1)

        ht_list = [state] * self.num_layers

        go = torch.zeros((source.shape[0], self.num_node, self.output_dim), device=source.device)
        out = []
        for t in range(self.horizon):
            state, ht_list = self.decoder(go, ht_list, [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1])
            go = self.proj(state)
            out.append(go)
            if self.training:     #这里的课程学习用了给予一定概率用真实值代替预测值来学习的教师-学生学习法（名字忘了，大概跟着有关）
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):  #如果满足条件，则用真实值代替预测值训练
                    go = traget[:, t, :, :self.input_dim]
        output = torch.stack(out, dim=1)


        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x
