import pytorch_lightning as pl
import torch
import numpy as np
 

class TrainModel(pl.LightningModule):
 
    def __init__(self, model, max_context_points: int=20):
        super().__init__()
        self.model = model
        self.max_context_points=max_context_points
 
        if torch.cuda.is_available():  
          self.dev = "cuda:0" 
        else:  
          self.dev = "cpu"
        self.my_log = []
        #https://github.com/cqql/neural-processes-pytorch/blob/998af68325bd81b6fcedfb327ebffaac5ff27f23/neural_process.py#L18
        self.autoenccrit = lambda mu, sigma, x: (x - mu)**2 / (2 * sigma**2) + torch.log(sigma)
        #self.autoenccrit = lambda mu, sigma, x: torch.nn.MSELoss()(mu, x)
        #self.autoenccrit = torch.nn.SmoothL1Loss(beta=0.1)
    
    def forward(self, xy, q):
        return self.model(xy, q)
 
    def training_step(self, batch, batch_idx):
        x, y = batch
        i_q = np.random.randint(2, self.max_context_points)
 
        xy = torch.cat([x[:, :i_q], y[:, :i_q]], -1)
        q = x#[:, :i_q*3]
        y_r = y#[:, :i_q*3]
        pred_mu, pred_sigma = self.model(xy, q)
        
        loss = torch.mean(self.autoenccrit(pred_mu, pred_sigma, y_r))
 
        ret =  {'loss': loss.detach()}
        self.my_log.append(ret)
        return loss
 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer