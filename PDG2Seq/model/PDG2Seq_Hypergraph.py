import torch
import torch.nn as nn
from model.PDG2SeqHyperCell import PDG2SeqHyperCell
import numpy as np

class PDG2Seq_HyperEncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1, 
                 dataset_name='NYC-Bike', hyperedge_types=['pick_drop', 'geo', 'temporal', 'correlation', 'pattern']):
        super(PDG2Seq_HyperEncoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        
        # First layer
        self.PDG2Seq_cells.append(PDG2SeqHyperCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, 
                                                   dataset_name, hyperedge_types))
        # Additional layers
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqHyperCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim,
                                                       dataset_name, hyperedge_types))

    def forward(self, x, init_state, node_embeddings=None):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.PDG2Seq_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.PDG2Seq_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class PDG2Seq_HyperDecoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1,
                 dataset_name='NYC-Bike', hyperedge_types=['pick_drop', 'geo', 'temporal', 'correlation', 'pattern']):
        super(PDG2Seq_HyperDecoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        
        # First layer
        self.PDG2Seq_cells.append(PDG2SeqHyperCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim,
                                                   dataset_name, hyperedge_types))
        # Additional layers
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqHyperCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim,
                                                       dataset_name, hyperedge_types))

    def forward(self, xt, init_state, node_embeddings=None):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        
        for i in range(self.num_layers):
            state = self.PDG2Seq_cells[i](current_inputs, init_state[i], node_embeddings)
            output_hidden.append(state)
            current_inputs = state
        
        return current_inputs, output_hidden


class PDG2Seq_Hypergraph(nn.Module):
    def __init__(self, args):
        super(PDG2Seq_Hypergraph, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = getattr(args, 'use_day', True)
        self.use_W = getattr(args, 'use_week', True)
        self.cl_decay_steps = args.lr_decay_step
        self.dataset_name = args.dataset
        
        # Hypergraph config
        self.use_hypergraph = getattr(args, 'use_hypergraph', True)
        self.hyperedge_types = getattr(args, 'hyperedge_types', ['pick_drop', 'geo', 'temporal', 'correlation', 'pattern'])
        if isinstance(self.hyperedge_types, str):
            self.hyperedge_types = self.hyperedge_types.split(',')
        
        # Node embeddings
        self.node_embeddings1 = nn.Parameter(torch.empty(self.num_node, args.embed_dim))
        self.T_i_D_emb1 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(7, args.time_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(7, args.time_dim))

        # Encoder và Decoder với hypergraph
        self.encoder = PDG2Seq_HyperEncoder(args.num_nodes, args.input_dim, args.rnn_units, 
                                            getattr(args, 'cheb_order', 2), args.embed_dim, args.time_dim, 
                                            args.num_layers, self.dataset_name, self.hyperedge_types)
        self.decoder = PDG2Seq_HyperDecoder(args.num_nodes, args.input_dim, args.rnn_units,
                                            getattr(args, 'cheb_order', 2), args.embed_dim, args.time_dim,
                                            args.num_layers, self.dataset_name, self.hyperedge_types)

        # Predictor
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, target=None, batches_seen=None):
        # source: B, T_1, N, D
        # target: B, T_2, N, D

        init_state = self.encoder.init_hidden(source.shape[0]).to(source.device)
        
        # Encoding
        output, encoder_state = self.encoder(source, init_state)
        
        # Decoding
        go_symbol = torch.zeros((source.shape[0], self.num_node, self.input_dim), device=source.device)
        decoder_input = go_symbol
        decoder_state = encoder_state
        outputs = []

        for t in range(self.horizon):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            decoder_input = self.proj(decoder_output)  # B, N, output_dim
            outputs.append(decoder_input)
            
            if target is not None and t < target.shape[1]:
                # Teacher forcing during training
                decoder_input = target[:, t, :, :]
            else:
                # Use prediction as next input
                decoder_input = decoder_input

        outputs = torch.stack(outputs, dim=1)  # B, T_2, N, output_dim
        return outputs
