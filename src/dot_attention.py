from .modules import weights_init, init_kernels

import torch
import math

class DotAttention(torch.nn.Module):
    """
    Multihead attention mechanism
    """
    def __init__(self, num_hidden, h, N=32):
        super(DotAttention, self).__init__()

        self.N = N
        self.num_hidden = num_hidden
        self.h = h
        self.num_hidden_per_attn = num_hidden // h
        self.bias = torch.nn.Parameter(torch.zeros(self.h))

        self.sm = torch.nn.Softmax()
    
    def reset_parameters(self):
        weights_init(self)
 
    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        
        key = key.view(batch_size, seq_k, self.h, 1)
        value = value.view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = query.view(batch_size, seq_q, self.h, 1)
 
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, 1)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, 1)

         
        q_times_k = torch.bmm(query.transpose(1,2), key)    
        q_times_k = self.sm(q_times_k)
        q_times_k_times_v = torch.bmm(q_times_k, value)
        
        result = q_times_k_times_v.view(self.h, batch_size, seq_q, self.num_hidden_per_attn).permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        return result
    
    def get_kernel_params(self):      
        imag_amps = self.imag_amps
        real_amps = self.real_amps

        return real_amps, imag_amps


    def get_freqs(self, N_kernel):
        step = 1 / (N_kernel -1 )
        #return torch.arange(0, 1 + step, step=step) * 2 *  math.pi
        return torch.arange(0, 1 + step, step=step) * N_kernel * math.pi #Test just half space* 2

    def viz_kernel_in_fourier_domain(self):
        import matplotlib.pyplot as plt

        kr, ki = self.get_kernel_params()
        kr = kr.view([self.h, -1])
        ki = ki.view([self.h, -1])

        plt.figure()
        for i in range(kr.size()[0]):
            plt.plot(kr[i].detach().numpy(), c='r', label='Re', marker='o')
            plt.plot(ki[i].detach().numpy(), c='b', label='Im', marker='o')
            plt.legend()

    def viz_kernel_in_spatial_domain(self):
        import matplotlib.pyplot as plt

        kr, ki = self.get_kernel_params()
  
        freq = self.get_freqs(kr.shape[-1])

        kr = kr.view([self.h, -1])
        ki = ki.view([self.h, -1])

        plt.figure()
        for i in range(kr.size()[0]):
            step = 0.01
            x = torch.arange(-1, 1+step, step=step)
            y = [torch.sum(torch.cos(freq*x[q])*kr[i] + torch.sin(freq*x[q])*ki[i]).detach().numpy() for q in range(x.shape[0])]

            plt.plot(x.detach().numpy(), y)        