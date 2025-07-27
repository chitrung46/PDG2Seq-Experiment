import torch
import torch.nn as nn
from model.PDG2Seq_DGCN import PDG2Seq_HyperGCN
from collections import OrderedDict
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FC, self).__init__()
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.mlp=nn.Sequential(
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, dim_out))]))

    def forward(self, x):
        ho = self.mlp(x)
        return ho

class PDG2SeqHyperCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, dataset_name='NYC-Bike', 
                 hyperedge_types=['pick_drop', 'geo', 'temporal', 'correlation', 'pattern']):
        super(PDG2SeqHyperCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.dataset_name = dataset_name
        self.hyperedge_types = hyperedge_types
        
        # Sử dụng HyperGCN thay vì GCN thường
        self.gate = PDG2Seq_HyperGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, time_dim, 
                                     dataset_name, hyperedge_types)
        self.update = PDG2Seq_HyperGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, time_dim,
                                       dataset_name, hyperedge_types)
        self.fc1 = FC(dim_in + self.hidden_dim, time_dim)
        self.fc2 = FC(dim_in + self.hidden_dim, time_dim)

    def forward(self, x, state, node_embeddings=None):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        
        # Hypergraph convolution cho gate và update
        gate_output = self.gate(input_and_state)  # B, N, 2*hidden_dim
        update_output = self.update(input_and_state)  # B, N, hidden_dim
        
        # Split gate output thành reset và update gates
        gate_r, gate_z = torch.split(gate_output, self.hidden_dim, dim=-1)
        gate_r = torch.sigmoid(gate_r)
        gate_z = torch.sigmoid(gate_z)
        
        # GRU update
        new_state = gate_z * state + (1 - gate_z) * torch.tanh(update_output * gate_r)
        
        return new_state

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
