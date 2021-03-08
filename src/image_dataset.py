import torch
import torchvision
import numpy as np
from typing import List
from PIL import Image

class NPImageDataset(torch.utils.data.Dataset):
    def __init__(self, files: List, height: int, width: int, s: int=100):
        super().__init__()
        self.files = files
        self.height, self.width = height, width
        self.s = s
        
        self.tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop((height, width))
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx:int):
        img_tensor = self.tfms(Image.open(self.files[idx]))
        h_coords = torch.randint(0, self.height, size=(self.s,))
        w_coords = torch.randint(0, self.width, size=(self.s,))
        
        xs = torch.stack([h_coords, w_coords], -1).float()
        ys = img_tensor[:,h_coords, w_coords].float()
        
        return xs, ys.transpose(1,0)
        