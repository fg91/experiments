from .modules import weights_init

import torch

class KernelAttention(torch.nn.Module):
    """
    Multihead attention mechanism
    """
    def __init__(self, num_hidden, h, N=32):
        super(KernelAttention, self).__init__()

        self.N = N
        self.num_hidden = num_hidden
        self.h = h
        self.num_hidden_per_attn = num_hidden // h
    
    def reset_parameters(self):
        weights_init(self)
 
    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        
        key = key.view(batch_size, seq_k, self.h, 1)
        value = value.view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = query.view(batch_size, seq_q, self.h, 1)
 
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)
         
        qk_cross = (key.unsqueeze(1)-query.unsqueeze(2)).view(batch_size, seq_k, seq_q, -1)
        q_times_k = self.kernel(qk_cross)
        q_times_k = q_times_k.view(batch_size, seq_k, seq_q, self.h).permute(3, 0, 1, 2).contiguous().view(-1,seq_k, seq_q)

        q_times_k_times_v = torch.bmm(q_times_k, value)
        
        result = q_times_k_times_v.view(self.h, batch_size, seq_q, self.num_hidden_per_attn).permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        return result    

    def viz_kernel_in_spatial_domain(self):
        #TODO
