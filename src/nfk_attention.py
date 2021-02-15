from .modules import weights_init, init_kernels

import torch
import math

def lspline_kernel(ax, ay, a_scale, xx):
  # construct exp bases for spline
  ax = ax*a_scale

  # select spline segment by constructing mask
  seg_sel = (xx.unsqueeze(-1) >= ax[:,:,:,:-1].unsqueeze(-2))*(xx.unsqueeze(-1) < ax[:,:,:,1:].unsqueeze(-2))

  # do linear pline interpolation for samples xx
  xy = ((ay[:,:,:,1:]-ay[:,:,:,:-1])/(ax[:,:,:,1:]-ax[:,:,:,:-1])).unsqueeze(-2) # calculate slope
  xy = ay[:,:,:,:-1].unsqueeze(-2) + (xx.unsqueeze(-1) - ax[:,:,:,:-1].unsqueeze(-2)) * xy # interpolate segments
  xy = (seg_sel*xy).sum(dim=[-1]) # apply mask

  return xy

class NormalizedFourierKernelAttention(torch.nn.Module):
    """
    Multihead attention mechanism
    """
    def __init__(self, num_hidden, h, N=32):
        super(NormalizedFourierKernelAttention, self).__init__()

        self.N = N
        self.num_hidden = num_hidden
        self.h = h
        self.num_hidden_per_attn = num_hidden // h
        self.bias = torch.nn.Parameter(torch.zeros(self.h))        
        self.real_amps = torch.nn.Parameter(init_kernels(self.h, self.N-1).clone().detach().requires_grad_(True))
        self.imag_amps = torch.nn.Parameter(torch.zeros(self.h, 1, 1, self.N-1, requires_grad=True))
        self.scale_amps = torch.nn.Parameter(torch.zeros(self.h, 1, 1, 1, requires_grad=True))

        step = 1./(self.real_amps.shape[-1])

        # construct amps x
        ax_slope = torch.tensor(math.pi*2)
        x_amps = torch.arange(0, 1+step, step=step)
        #x_amps = ((torch.exp(x_amps*ax_slope)-1)/(torch.exp(ax_slope)-1))
        self.x_amps = torch.nn.Parameter(x_amps.view(1,1,1,-1), requires_grad=False)

        self.freqs_x = torch.nn.Parameter(torch.arange(0, 1+step, step=step).view(1,1,1,-1), requires_grad=False)
        self.freqs = torch.nn.Parameter(self.get_freqs(self.N).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([self.h, 1, 1, 1]), requires_grad=False)
    
    def reset_parameters(self):
        weights_init(self)
 
    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        
        key = key.view(batch_size, seq_k, self.h, 1)
        value = value.view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = query.view(batch_size, seq_q, self.h, 1)
 
        key = key.permute(2, 0, 1, 3).contiguous()
        query = query.permute(2, 0, 1, 3).contiguous()
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)

        # norm q, k     
        max_qk = torch.amax(torch.abs(torch.cat([key, query], -2)), -2, keepdim=True)
        key = key/max_qk
        query = query/max_qk

         # do conv in freq domain
        freqs = self.freqs
        real_amps, imag_amps, scale_amps = self.get_kernel_params()

        # interpolate real/imag based on max_qk        
        xx = self.freqs_x * max_qk
        real_amps = lspline_kernel(self.x_amps, real_amps, scale_amps, xx)
        imag_amps = lspline_kernel(self.x_amps, imag_amps, scale_amps, xx)
        #real_amps = real_amps / max_qk
        #imag_amps = imag_amps / max_qk
        
        query_prime = freqs * query
        key_prime = freqs * key
        
        sin_query_prime = torch.sin(query_prime)
        cos_query_prime = torch.cos(query_prime)
        sin_key_prime   = torch.sin(key_prime)
        cos_key_prime   = torch.cos(key_prime)

        
        trig_query = torch.cat([
            real_amps * cos_query_prime,
            real_amps * sin_query_prime,
            imag_amps * sin_query_prime,
            -imag_amps* cos_query_prime,
        ], -1)
        
        trig_key = torch.cat([
            cos_key_prime,
            sin_key_prime,
            cos_key_prime,
            sin_key_prime,
        ], -1)

        trig_query = trig_query.view(-1, seq_q, self.N*4)
        trig_key = trig_key.view(-1, seq_k, self.N*4)

        k_times_v = torch.bmm(trig_key.transpose(1,2), value)
        
        q_times_k_times_v = torch.bmm(trig_query, k_times_v)
        
        result = q_times_k_times_v.view(self.h, batch_size, seq_q, self.num_hidden_per_attn).permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        return result
    
    def get_kernel_params(self):      
        imag_amps = self.imag_amps
        real_amps = self.real_amps
        
        # pad since last frequency should be always 0
        pad = (0, 1)
        real_amps = torch.nn.functional.pad(real_amps, pad, mode='constant', value=0)
        imag_amps = torch.nn.functional.pad(imag_amps, pad, mode='constant', value=0)
        scale_amps = torch.nn.functional.softplus(self.scale_amps)+0.5 # make always positive

        return real_amps, imag_amps, scale_amps


    def get_freqs(self, N_kernel):
        step = 1 / (N_kernel -1 )
        #return torch.arange(0, 1 + step, step=step) * 2 *  math.pi
        return torch.arange(0, 1 + step, step=step) * N_kernel * math.pi #Test just half space* 2

    def viz_kernel_in_fourier_domain(self):
        import matplotlib.pyplot as plt

        kr, ki, ks = self.get_kernel_params()
        kr = kr.view([self.h, -1])
        ki = ki.view([self.h, -1])
        kx = (self.x_amps*ks).view([self.h, -1])

        plt.figure()
        for i in range(kr.size()[0]):
            plt.plot(kx[i].detach().numpy(), kr[i].detach().numpy(), c='r', label='Re', marker='o')
            plt.plot(kx[i].detach().numpy(), ki[i].detach().numpy(), c='b', label='Im', marker='o')
            plt.legend()

    def viz_kernel_in_spatial_domain(self):
        import matplotlib.pyplot as plt

        kr, ki, ks = self.get_kernel_params()
  
        step = 1./(kr.shape[-1]-1)
        xx = torch.arange(0, 1+step, step=step).view(1,1,1,-1)
        kr = lspline_kernel(self.x_amps, kr, ks, xx)
        ki = lspline_kernel(self.x_amps, ki, ks, xx)

        freq = self.get_freqs(kr.shape[-1])

        kr = kr.view([self.h, -1])
        ki = ki.view([self.h, -1])

        print(ks[:,-1])

        plt.figure()
        for i in range(kr.size()[0]):
            step = 0.01
            x = torch.arange(-1, 1+step, step=step)
            y = [torch.sum(torch.cos(freq*x[q])*kr[i] + torch.sin(freq*x[q])*ki[i]).detach().numpy() for q in range(x.shape[0])]

            plt.plot(x.detach().numpy(), y)