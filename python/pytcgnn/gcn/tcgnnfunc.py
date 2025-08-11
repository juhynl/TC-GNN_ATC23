import torch
import TCGNN

class TCGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # GEMM node update
        X_prime = torch.mm(X, weights)
        
        # X_prime_t = torch.ones_like(X_prime)
        # X_prime_t = gen_test_tensor(X_prime)
        # print("=========Before AggreAGNNion========")
        # print(X_prime_t)
        # sys.exit(0)

        # SpMM: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
        # print("==========After Aggreation=========")
        # print(X_prime)
        # sys.exit(0)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # SPMM backward propaAGNNion.
        d_input_prime = TCGNN.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        return d_input, d_weights, None, None, None, None, None, None