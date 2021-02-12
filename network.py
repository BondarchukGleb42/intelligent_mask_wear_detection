import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Softmax_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        e = torch.exp(x - x.max(1, True)[0] )
        summ = e.sum(1, True)[0]
        return e / summ

class Flatten(torch.nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)


conv_net = models.resnet18(pretrained=True)
conv_net.fc = nn.Linear(512, 2)

class MaskDetector(nn.Module):
    def __init__(self, device='cpu', model=conv_net):
        super(MaskDetector, self).__init__()
        self.device = device
        self.model = model

        PATH = 'trained_weights.pt'
        self.load_state_dict(torch.load(PATH))

    def predict(self, frame):

        self.eval()
        self.to(self.device)

        frame = np.expand_dims(frame, 0)
        frame = torch.FloatTensor(frame)
        frame = frame.to(self.device)
        with torch.no_grad():
            probs = self.model(frame)

        return probs
