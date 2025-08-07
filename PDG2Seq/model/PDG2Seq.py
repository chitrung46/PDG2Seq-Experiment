import torch
import torch.nn as nn
from model.PDG2SeqCell import PDG2SeqCell
import numpy as np
from .revin import RevIN
from .patch_embedding import PatchEmbedding

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

        self.use_revin = getattr(args, 'use_revin', True)
        if self.use_revin == True:
            print("use revin")
            self.revin = RevIN(
                num_features=self.input_dim,
                affine=getattr(args, 'revin_affine', True),
                subtract_last=getattr(args, 'revin_subtract_last', False)
            )

        self.use_patch = getattr(args, 'use_patch', False)
        self.patch_len = getattr(args, 'patch_len', 8)
        self.patch_stride = getattr(args, 'patch_stride', 4)
        # Nếu muốn dùng patch embedding, thêm dòng này:
        # self.patch_embed_dim = getattr(args, 'patch_embed_dim', 16)
        # self.patch_embedding = PatchEmbedding(self.patch_len, self.patch_stride, self.input_dim, self.patch_embed_dim)

        if self.use_patch == True:
            self.encoder_input_dim = self.patch_len * self.input_dim
        else:
            self.encoder_input_dim = self.input_dim

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
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, traget=None, batches_seen=None):
        # source: B, T, N, D
        # traget: B, T, N, D

        if self.revin == True:
            B, T, N, D = source.shape
            source_main = source[..., :self.input_dim]
            source_main = source_main.permute(0, 2, 1, 3).reshape(B * N, T, self.input_dim)
            source_main = self.revin(source_main, 'norm')
            source_main = source_main.reshape(B, N, T, self.input_dim).permute(0, 2, 1, 3)
            source = torch.cat([source_main, source[..., self.input_dim:]], dim=-1)

        if self.use_patch == True:
            # Chia patch cho 2 đặc trưng đầu, giảm chiều dài chuỗi
            B, T, N, D = source.shape
            x_patch = source[..., :self.input_dim]  # (B, T, N, 2)
            x_patch = x_patch.permute(0, 2, 1, 3).reshape(B * N, T, self.input_dim)
            patches = x_patch.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)  # (B*N, num_patches, patch_len, 2)
            num_patches = patches.shape[1]
            patches = patches.reshape(B, N, num_patches, self.patch_len * self.input_dim)  # (B, N, num_patches, patch_len*2)
            patches = patches.permute(0, 2, 1, 3)  # (B, num_patches, N, patch_len*2)
            source_for_encoder = patches
        else:
            source_for_encoder = source[..., :self.input_dim]  # (B, T, N, 2)

        # Node embedding và các bước khác giữ nguyên
        t_i_d_data1 = source[..., 0, -2]
        t_i_d_data2 = traget[..., 0, -2]
        T_i_D_emb1_en = self.T_i_D_emb1[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb2_en = self.T_i_D_emb2[(t_i_d_data1 * 288).type(torch.LongTensor)]

        T_i_D_emb1_de = self.T_i_D_emb1[(t_i_d_data2 * 288).type(torch.LongTensor)]
        T_i_D_emb2_de = self.T_i_D_emb2[(t_i_d_data2 * 288).type(torch.LongTensor)]
        if self.use_W:
            d_i_w_data1 = source[..., 0, -1]
            d_i_w_data2 = traget[..., 0, -1]
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

        en_node_embeddings = [node_embedding_en1, node_embedding_en2, self.node_embeddings1]

        # Truyền vào encoder
        init_state = self.encoder.init_hidden(source_for_encoder.shape[0]).to(source.device)
        state, _ = self.encoder(source_for_encoder, init_state, en_node_embeddings)
        state = state[:, -1:, :, :].squeeze(1)

        ht_list = [state] * self.num_layers

        go = torch.zeros((source.shape[0], self.num_node, self.output_dim), device=source.device)
        out = []
        for t in range(self.horizon):
            state, ht_list = self.decoder(go, ht_list, [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1])
            go = self.proj(state)
            out.append(go)
            if self.training:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    go = traget[:, t, :, :self.input_dim]
        output = torch.stack(out, dim=1)

        if self.revin == True:
            B, T, N, D = output.shape
            output_main = output[..., :self.input_dim]
            output_main = output_main.permute(0, 2, 1, 3).reshape(B * N, T, 2)
            output_main = self.revin(output_main, 'denorm')
            output_main = output_main.reshape(B, N, T, 2).permute(0, 2, 1, 3)
            output = torch.cat([output_main, output[..., self.output_dim:]], dim=-1) if output.shape[-1] > 2 else output_main
        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x