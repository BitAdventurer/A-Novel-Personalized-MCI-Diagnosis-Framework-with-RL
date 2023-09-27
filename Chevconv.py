import torch
import torch.nn as nn
import math
import gcn_util

def chev_conv(self, x, adj_orig):
    # A` * X * W
    torch.cuda.empty_cache()
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = gcn_util.laplacian_norm_adj(adj_orig)
    adj = gcn_util.add_self_loop(-adj)
    Tx_0 = x[0]
    Tx_1 = x[0]  # Dummy.

    out = torch.matmul(Tx_0, self.weight[0])
    # propagate_type: (x: Tensor, norm: Tensor)

    if self.weight.shape[0] > 1: 
        Tx_1 = torch.matmul(adj, x)
        out = out + torch.matmul(Tx_1, self.weight[1])
        
    for k in range(2, self.weight.shape[0]):
        Tx_2 = torch.matmul(adj, Tx_1)
        Tx_2 = 2. * Tx_2 - Tx_0
        out = out + torch.matmul(Tx_1, self.weight[k])
        Tx_0, Tx_1 = Tx_1, Tx_2
    if self.bias is not None:
        out += self.bias

    return out