import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import LinearInterval, Conv2dInterval


class Large(nn.Module):

    def __init__(self, eps=0):
        super().__init__()
        self.conv1 = Conv2dInterval(3, 64, 3, 1, eps=eps, input_layer=True)
        self.conv2 = Conv2dInterval(64, 64, 3, 1, eps=eps)
        self.conv3 = Conv2dInterval(64, 128, 3, 2, eps=eps)
        self.conv4 = Conv2dInterval(128, 128, 3, 1, eps=eps)
        self.conv5 = Conv2dInterval(128, 128, 3, 1, eps=eps)
        self.fc1 = LinearInterval(128 * 9 * 9, 200, eps=eps)
        self.last = LinearInterval(200, 10, eps=eps)

        self.a = nn.Parameter(torch.zeros(7), requires_grad=True)
        self.e = None

        self.eps = eps
        self.bounds = None

    def save_bounds(self, x):
        s = x.size(1) // 3
        self.bounds = x[:, s:2 * s], x[:, 2 * s:]

    def calc_eps(self, r):
        exp = self.a.exp()
        self.e = r * exp / exp.sum()

    def set_eps(self, eps):
        self.calc_eps(eps)

        self.conv1.eps = self.e[0]
        self.conv2.eps = self.e[1]
        self.conv3.eps = self.e[2]
        self.conv4.eps = self.e[3]
        self.conv5.eps = self.e[4]

        self.fc1.eps = self.e[5]
        for _, layer in self.last.items():
            layer.eps = self.e[6]

    def features(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        self.save_bounds(x)
        return x

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return {k: v[:, :v.size(1) // 3] for k, v in x.items()}


def large():
    return Large()
