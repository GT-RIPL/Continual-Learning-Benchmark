import torch
import torch.nn as nn
import torch.nn.functional as f
from interval.layers import LinearInterval, Conv2dInterval, MaxPool2dInterval, AvgPool2dInterval, IntervalDropout


class CNN(nn.Module):

    def __init__(self, in_channel=3, out_dim=10, pooling=nn.MaxPool2d):
        super().__init__()

        self.input = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.c1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            nn.Dropout(0.25)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            nn.Dropout(0.25)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=1),
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*5*5, 256),
            nn.ReLU()
        )
        self.last = nn.Linear(256, out_dim)

    def features(self, x):
        x = self.input(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def cnn():
    return CNN()


def cnn_avg():
    return CNN(pooling=nn.AvgPool2d)


class IntervalCNN(nn.Module):

    def __init__(self, in_channel=3, out_dim=10, pooling=MaxPool2dInterval, eps=0):
        super(IntervalCNN, self).__init__()

        self.input = Conv2dInterval(in_channel, 32, kernel_size=3, stride=1, padding=1, input_layer=True)
        self.c1 = nn.Sequential(
            Conv2dInterval(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Conv2dInterval(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25)
        )
        self.c2 = nn.Sequential(
            Conv2dInterval(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Conv2dInterval(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25)
        )
        self.c3 = nn.Sequential(
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            pooling(2, stride=2, padding=1),
            IntervalDropout(0.25)
        )
        self.fc1 = nn.Sequential(
            LinearInterval(128 * 5 * 5, 256),
            nn.ReLU()
        )
        self.last = LinearInterval(256, out_dim)
        self.a = nn.Parameter(torch.zeros(9), requires_grad=True)
        self.e = torch.zeros(9)

        self.bounds = None

    def save_bounds(self, x):
        s = x.size(1) // 3
        self.bounds = x[:, s:2*s], x[:, 2*s:]

    def calc_eps(self, r):
        exp = self.a.exp()
        self.e = r * exp / exp.sum()

    def print_eps(self, head="All"):
        for c in self.children():
            if isinstance(c, nn.Sequential):
                for layer in c.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        e = layer.eps.detach()
                        print(f"sum: {e.sum()} - mean: {e.mean()} - std: {e.std()}")
                        print(f" * min {e.min()}, max: {e.max()}")
            elif isinstance(c, nn.ModuleDict):
                e = c[head].eps.detach()
                print(f"sum: {e.sum()} - mean: {e.mean()} - std: {e.std()}")
                print(f" * min {e.min()}, max: {e.max()}")
            elif isinstance(c, (Conv2dInterval, LinearInterval)):
                e = c.eps.detach()
                print(f"sum: {e.sum()} - mean: {e.mean()} - std: {e.std()}")
                print(f" * min {e.min()}, max: {e.max()}")

    def reset_importance(self):
        for c in self.children():
            if isinstance(c, nn.Sequential):
                for layer in c.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        layer.rest_importance()
            elif isinstance(c, nn.ModuleDict) and "All" in c.keys():
                c["All"].rest_importance()
            elif isinstance(c, (Conv2dInterval, LinearInterval)):
                c.rest_importance()

    def set_eps(self, eps, trainable=False, head="All"):
        if trainable:
            self.calc_eps(eps)
        else:
            self.e[:] = eps
        i = 0
        for c in self.children():
            if isinstance(c, nn.Sequential):
                for layer in c.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        layer.calc_eps(self.e[i])
                        i += 1
            elif isinstance(c, nn.ModuleDict):
                self.last[head].calc_eps(eps)
                i += 1
            elif isinstance(c, (Conv2dInterval, LinearInterval)):
                c.calc_eps(self.e[i])
                i += 1

    def features(self, x):
        x = self.input(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        self.save_bounds(x)
        return x

    def logits(self, x):
        return self.last(x)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return {k: v[:, :v.size(1)//3] for k, v in x.items()}


def interval_cnn():
    return IntervalCNN()


def interval_cnn_avg():
    return IntervalCNN(pooling=AvgPool2dInterval)


if __name__ == '__main__':
    cnn = IntervalCNN()
    x = torch.randn(12, 3, 32, 32)
    cnn(x)
