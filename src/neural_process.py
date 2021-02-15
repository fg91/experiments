from .modules import MLP, weights_init
from .nfk_attention import NormalizedFourierKernelAttention

import torch

class AttentionNetwork(torch.nn.Module):
    """
    Attention Network
    """
    def __init__(self, num_hidden, kq_num_hidden=1, h=8, attention_mechanism=NormalizedFourierKernelAttention):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(AttentionNetwork, self).__init__()
 
        self.num_hidden = num_hidden
        self.h = h

        self.key = torch.nn.Linear(kq_num_hidden, self.h, bias=False)
        self.value = torch.nn.Linear(num_hidden, num_hidden, bias=False)
        self.query = torch.nn.Linear(kq_num_hidden, self.h, bias=False)
 
        self.multihead = attention_mechanism(num_hidden, self.h)
        self.final_linear = torch.nn.Linear(num_hidden, num_hidden) 
        self.layer_norm1 = torch.nn.LayerNorm(num_hidden)
        self.mlp = MLP(num_hidden, num_hidden, n_hidden_layers=2)
        self.layer_norm2 = torch.nn.LayerNorm(num_hidden)
  
    def reset_parameters(self):
        weights_init(self)
 
    def forward(self, key, query, value, residual):
            batch_size = key.size(0)
            seq_k = key.size(1)
            seq_q = query.size(1)
            
            key = self.key(key)
            value = self.value(value)
            query = self.query(query)
 
            # Get context vector
            result = self.multihead(key, value, query)
            result = self.final_linear(result)
            result = self.layer_norm1(residual + result)
            
            # FC
            result = result + self.mlp(result)
            result = self.layer_norm2(result)

            return result        
 
class NP(torch.nn.Module):
    def __init__(self, indim=1, outdim=1):
        #batch token channel
        super(NP, self).__init__()
        self.outdim = outdim
        self.indim = indim
        self.h = 64
        n_attention = 4
 
        self.enc_xy = MLP((indim+outdim), self.h)        
        self.enc_keyself = MLP(self.indim, self.h)        
        #self.enc_keysz = MLP(self.indim, self.h)
        self.enc_keycross = MLP(self.indim, self.h)
        self.enc_q = MLP(self.indim, self.h)
        #self.enc_target = MLP(self.indim, self.h, hidden_size=self.h)
        #self.z_attentions = torch.nn.ModuleList([AttentionNetwork(self.h) for _ in range(n_attention)])
        self.self_attentions = torch.nn.ModuleList([AttentionNetwork(self.h, kq_num_hidden=self.h) for _ in range(n_attention)])
        self.cross_attentions = torch.nn.ModuleList([AttentionNetwork(self.h, kq_num_hidden=self.h) for _ in range(n_attention)])
        #self.penultimate_layer_presum = MLP(self.h, self.h, n_hidden_layers=2)
 
        #bottleneck_h = int(self.h * self.num_z / 2) #*numin
        #self.dec = MLP(self.h * self.num_z, outdim * self.num_z, bottleneck_h, n_hidden_layers=2)
        self.dec = MLP(self.h, outdim*2, self.h, n_hidden_layers=2)
      
    def forward(self, xy, q):
        x = xy[:,:,:self.indim]
        enc = enc_o = self.enc_xy(xy)
        '''
        # context
        keyself_enc = self.enc_keyself(x)
        for attention in self.self_attentions:
            enc = attention(keyself_enc, enc_o, enc, enc)
        '''
        # z
        '''
        keyz_enc = x #self.enc_keysz(x)
        for attention in self.z_attentions:
            enc_z, _ = attention(keyz_enc, x, enc_z, enc_z)
        enc_z = torch.mean(enc_z, 1, keepdim=True)
        enc_z = self.penultimate_layer_presum(enc_z)
        enc_z = enc_z.repeat(1, q.size()[1], 1)
        '''
        
        # query
        keycross_enc = self.enc_keycross(x)
        q_enc = q_enc_o = self.enc_q(q)
        for attention in self.cross_attentions:
            q_enc = attention(keycross_enc, q_enc_o, enc, q_enc)
 
        #q_dec = torch.cat([q_enc, enc_z], -1) #, enc_z
        q_dec = q_enc
 
        # dec
        #q_dec = q_dec.view([int(x.size()[0]/self.num_z), q.size()[1], self.h*self.num_z])
        ret = self.dec(q_dec)
        #ret = ret.view([-1, q.size()[1], self.outdim])
 
        mu = ret[:,:,:self.outdim]
        sigma = torch.nn.functional.softplus(ret[:,:,self.outdim:])
        #ret = torch.distributions.Normal(mu, sigma).rsample()
        return mu, sigma
 
    def reset_parameters(self):
        weights_init(self)