import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def MLP100():
    return MLP(hidden_dim=100)


def MLP400():
    return MLP(hidden_dim=400)


def MLP1000():
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)