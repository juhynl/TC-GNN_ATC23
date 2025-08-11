import torch
from .tcgnnfunc import TCGNNFunction
from .graph import GCNGraph
import math

class GraphConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, g: GCNGraph, x: torch.Tensor):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        
        return TCGNNFunction.apply(x, self.weights, g.row_pointers, g.column_index, g.blockPartition, g.edgeToColumn, g.edgeToRow)