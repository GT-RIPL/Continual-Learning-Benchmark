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

        self.input = Conv2dInterval(in_channel, 32, kernel_size=3, stride=1, padding=1, eps=eps, input_layer=True)
        self.c1 = nn.Sequential(
            Conv2dInterval(32, 32, kernel_size=3, stride=1, padding=1, eps=eps),
            nn.ReLU(),
            Conv2dInterval(32, 64, kernel_size=3, stride=1, padding=1, eps=eps),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25)
        )
        self.c2 = nn.Sequential(
            Conv2dInterval(64, 64, kernel_size=3, stride=1, padding=1, eps=eps),
            nn.ReLU(),
            Conv2dInterval(64, 128, kernel_size=3, stride=1, padding=1, eps=eps),
            nn.ReLU(),
            pooling(2, stride=2, padding=0),
            IntervalDropout(0.25)
        )
        self.c3 = nn.Sequential(
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1, eps=eps),
            nn.ReLU(),
            Conv2dInterval(128, 128, kernel_size=3, stride=1, padding=1, eps=eps),
            nn.ReLU(),
            pooling(2, stride=2, padding=1),
            IntervalDropout(0.25)
        )
        self.fc1 = nn.Sequential(
            LinearInterval(128 * 5 * 5, 256, eps=eps),
            nn.ReLU()
        )
        self.last = LinearInterval(256, out_dim, eps=eps)
        self.a = nn.Parameter(torch.zeros(9), requires_grad=True)
        self.e = None

        self.bounds = None

    def save_bounds(self, x):
        s = x.size(1) // 3
        self.bounds = x[:, s:2*s], x[:, 2*s:]

    def calc_eps(self, r):
        exp = self.a.exp()
        self.e = r * exp / exp.sum()

    def set_eps(self, eps, trainable=False):
        if trainable:
            self.calc_eps(eps)

            self.input.eps = self.e[0]
            self.c1[0].eps = self.e[1]
            self.c1[2].eps = self.e[2]
            self.c2[0].eps = self.e[3]
            self.c2[2].eps = self.e[4]
            self.c3[0].eps = self.e[5]
            self.c3[2].eps = self.e[6]
            self.fc1[0].eps = self.e[7]
            for _, layer in self.last.items():
                layer.eps = self.e[8]
        else:
            # self.input.eps = eps
            # self.c1[0].eps = eps
            # self.c1[2].eps = eps
            # self.c2[0].eps = eps
            # self.c2[2].eps = eps
            # self.c3[0].eps = eps
            # self.c3[2].eps = eps
            # self.fc1[0].eps = eps
            for _, layer in self.last.items():
                layer.eps = eps

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
