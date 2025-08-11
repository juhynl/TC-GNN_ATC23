import torch
from torch_geometric.utils import add_self_loops, get_laplacian
import warnings
import numpy as np
from scipy import sparse as sp

import TCGNN

class GCNGraph:
    def __init__(
        self,
        column_index: torch.Tensor, row_pointers:torch.Tensor, 
        num_nodes: int, 
        BLK_H: int, BLK_W: int, 
        blockPartition: torch.Tensor, 
        edgeToColumn: torch.Tensor, edgeToRow: torch.Tensor
    ):
        self.column_index = column_index
        self.row_pointers = row_pointers
        self.num_nodes = num_nodes
        self.BLK_H = BLK_H
        self.BLK_W = BLK_W
        self.blockPartition = blockPartition
        self.edgeToColumn = edgeToColumn
        self.edgeToRow = edgeToRow
    
    def to(self, device: torch.device):
        self.column_index = self.column_index.to(device)
        self.row_pointers = self.row_pointers.to(device)
        self.blockPartition = self.blockPartition.to(device)
        self.edgeToColumn = self.edgeToColumn.to(device)
        self.edgeToRow = self.edgeToRow.to(device)
        return self
    
    def __str__(self):
        return f"GCNGraph(row_pointers={self.row_pointers.shape}, column_index={self.column_index.shape}, degrees={self.degrees.shape}, t_window_rowTensor={self.t_window_rowTensor.shape}, t_atomicTensor={self.t_atomicTensor.shape})"

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0

def graph(edge_index: torch.Tensor, num_nodes: int):
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a tensor of shape (2, num_edges).")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")
    
    # tilde_edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # edge_weight = torch.ones(tilde_edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
    # row, col = tilde_edge_index
    
    # deg = torch.zeros(num_nodes, device=edge_index.device).scatter_add_(0, row, edge_weight)
    
    # deg_inv_sqrt = deg.pow_(-0.5)
    
    # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    # norm_adj_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # adj_coo = torch.sparse_coo_tensor(
    #     indices=tilde_edge_index,
    #     values=norm_adj_edge_weight,
    #     size=(num_nodes, num_nodes),
    #     device=edge_index.device
    # ).coalesce()

    adj_coo = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.shape[1]),
        size=(num_nodes, num_nodes),
        device=edge_index.device
    ).coalesce()
    
    with warnings.catch_warnings():
        # Filter warnings based on the message content.
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        adj_csr = adj_coo.to_sparse_csr()
    
    BLK_H = 16
    BLK_W = 8
    
    column_index = adj_csr.col_indices().to(torch.int)
    row_pointers = adj_csr.crow_indices().to(torch.int)
    
    num_edges = edge_index.size(1)

    num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
    blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
    
    edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
    edgeToRow = torch.zeros(num_edges, dtype=torch.int)
    
    TCGNN.preprocess(column_index, row_pointers, num_nodes, BLK_H, BLK_W, blockPartition, edgeToColumn, edgeToRow)
    
    g = GCNGraph(column_index, row_pointers, num_nodes, BLK_H, BLK_W, blockPartition, edgeToColumn, edgeToRow)
    
    return g