import torch
import numpy as np

def sinc(x, a=20):
  return np.sin(-(x-a)*a*20)/(-(x-a)*a*20)
  #return np.sin(-(x)*a*20)/(-(x)*a*20)
 
def sample_ds(n=10):
  x = (np.random.rand(n)-0.5)*2
  a = np.random.rand()
  return x.astype(np.float32), sinc(x, a).astype(np.float32), a

class SincDataset(torch.utils.data.Dataset):
  def __init__(self, N=10000, s=20):
        self.N = N
        self.s = s
 
  def __len__(self):
        return self.N
 
  def __getitem__(self, index):
        x,y,a = sample_ds(self.s)
        x = np.reshape(x, [-1, 1])
        y = np.reshape(y, [-1, 1])
        return x, y