import torch
import torch.nn as nn
import math
import gcn_util

def chev_conv(self, x, adj_orig):
    torch.cuda.empty_cache()  # Clearing CUDA cache, if necessary
    x = x.unsqueeze(0) if x.dim() == 2 else x  # Ensure the input has three dimensions
    
    adj = gcn_util.laplacian_norm_adj(adj_orig)  # Compute normalized Laplacian
    adj = gcn_util.add_self_loop(-adj)  # Add self-loop to adjacency matrix
    Tx_0 = x[0]
    
    # Perform the convolution
    out = torch.matmul(Tx_0, self.weight[0])
    
    if self.weight.shape[0] > 1:  # If more than one weight matrix is present
        Tx_1 = torch.matmul(adj, x)  # Compute the first transformation
        out += torch.matmul(Tx_1, self.weight[1])  # Update the output
    
    # Compute the subsequent transformations if more weight matrices are present
    for k in range(2, self.weight.shape[0]):
        Tx_2 = 2 * torch.matmul(adj, Tx_1) - Tx_0
        out += torch.matmul(Tx_1, self.weight[k])
        Tx_0, Tx_1 = Tx_1, Tx_2  # Update Tx_0 and Tx_1 for the next iteration
    
    if self.bias is not None:
        out += self.bias  # Add the bias if it is not None
    
    return out
