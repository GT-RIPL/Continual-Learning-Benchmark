import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import LinearInterval


class IntervalMLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(IntervalMLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.fc1 = LinearInterval(self.in_dim, hidden_dim, input_layer=True)
        self.fc2 = LinearInterval(hidden_dim, hidden_dim)
        # Subject to be replaced dependent on task
        self.last = LinearInterval(hidden_dim, out_dim)
        self.a = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.e = None

        self.bounds = None

    def save_bounds(self, x):
        s = x.size(1) // 3
        self.bounds = x[:, s:2*s], x[:, 2*s:]

    def calc_eps(self, r):
        exp = self.a.exp()
        self.e = r * exp / exp.sum()

    def print_eps(self):
        e1 = self.fc1.eps.detach()
        e2 = self.fc2.eps.detach()
        print(f"sum: {e1.sum()} - mean: {e1.mean()} - std: {e1.std()}")
        print(f"sum: {e2.sum()} - mean: {e2.mean()} - std: {e2.std()}")

        for name, layer in self.last.items():
            l = layer.eps.detach()
            print(f"last-{name} sum: {l.sum()} - mean: {l.mean()} - std: {l.std()}")
            layer.rest_importance()
        self.fc1.rest_importance()
        self.fc2.rest_importance()

    def reset_importance(self):
        for _, layer in self.last.items():
            layer.rest_importance()
        self.fc1.rest_importance()
        self.fc2.rest_importance()

    def set_eps(self, eps, trainable=False):
        if trainable:
            self.calc_eps(eps)
            self.fc1.calc_eps(self.e[0])
            self.fc2.calc_eps(self.e[1])
            for _, layer in self.last.items():
                layer.calc_eps(self.e[2])
        else:
            self.fc1.calc_eps(eps)
            self.fc2.calc_eps(eps)
            for _, layer in self.last.items():
                layer.calc_eps(eps)

    def features(self, x):
        x = x.view(-1, self.in_dim)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        self.save_bounds(x)
        return x

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return {k: v[:, :v.size(1)//3] for k, v in x.items()}


def interval_mlp400():
    return IntervalMLP(hidden_dim=400)


class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1, self.in_dim))
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