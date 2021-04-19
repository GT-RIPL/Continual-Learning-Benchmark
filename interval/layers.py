import torch
import torch.nn as nn
import torch.nn.functional as f


def split_activation(x):
    s = x.size(1) // 3
    mid = x[:, :s]
    low = x[:, s:2 * s]
    upp = x[:, 2 * s:]
    return mid, low, upp


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    return flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)


class AvgPool2dInterval(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_middle, x_lower, x_upper = split_activation(x)
        mid = super().forward(x_middle)
        lower = super().forward(x_lower)
        upper = super().forward(x_upper)
        return torch.cat((mid, lower, upper), dim=1)


class MaxPool2dInterval(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool2dInterval, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        x_middle, x_lower, x_upper = split_activation(x)
        mid = super().forward(x_middle)
        lower = super().forward(x_lower)
        upper = super().forward(x_upper)

        # mid, ini = super().forward(x_middle)
        # lower = retrieve_elements_from_indices(x_lower, ini)
        # upper = retrieve_elements_from_indices(x_upper, ini)
        return torch.cat((mid, lower, upper), dim=1)


class IntervalDropout(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1. / (1 - self.p)

    def forward(self, x):
        if self.training:
            x_mid, x_low, x_upp = split_activation(x.clone())
            mask = torch.bernoulli(self.p * torch.ones_like(x_mid)).long()
            x_mid[mask == 1] = 0.
            x_low[mask == 1] = 0.
            x_upp[mask == 1] = 0.
            return torch.cat((x_mid, x_low, x_upp), dim=1) * self.scale
        else:
            return x


class LinearInterval(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, input_layer=False):
        super().__init__(in_features, out_features, bias)
        self.importance = nn.Parameter(torch.zeros(self.weight.size()), requires_grad=True)
        # self.importance = nn.Parameter(torch.randn(self.weight.size()), requires_grad=True)
        self.eps = 0
        self.input_layer = input_layer

    def calc_eps(self, r):
        exp = self.importance.exp()
        self.eps = r * exp / exp.sum()
        self.eps = r * exp / exp.sum(dim=1)[:, None]
        # self.eps = r * exp / exp.sum(dim=0)[None, :]

    def rest_importance(self):
        # pass
        w1 = torch.abs(1 / self.eps)
        # self.importance.data = w1 / w1.sum()
        # self.importance.data = w1 / w1.sum(dim=1)[:, None]
        # self.importance.data = w1 / w1.sum(dim=1)[:, None]
        self.importance.data = torch.zeros(self.weight.size()).cuda()
        # self.importance.data = torch.randn(self.weight.size()).cuda()

    def forward(self, x):
        if self.input_layer:
            x = torch.cat((x, x, x), dim=1)

        x_middle, x_lower, x_upper = split_activation(x)

        middle = super().forward(x_middle)

        w_lower_pos = (self.weight - self.eps).clamp(min=0).t()
        w_lower_neg = (self.weight - self.eps).clamp(max=0).t()
        w_upper_pos = (self.weight + self.eps).clamp(min=0).t()
        w_upper_neg = (self.weight + self.eps).clamp(max=0).t()

        lower = x_lower @ w_lower_pos + x_upper @ w_lower_neg #+ self.bias
        upper = x_upper @ w_upper_pos + x_lower @ w_upper_neg #+ self.bias

        # w_lower = (self.weight - self.eps).t()
        # w_upper = (self.weight + self.eps).t()
        #
        # low_low = x_lower @ w_lower
        # low_upp = x_lower @ w_upper
        # upp_low = x_upper @ w_lower
        # upp_upp = x_upper @ w_upper
        #
        # min1, min2 = torch.min(low_low, low_upp), torch.min(upp_low, upp_upp)
        # max1, max2 = torch.max(low_low, low_upp), torch.max(upp_low, upp_upp)
        # lower = torch.min(min1, min2)  # + self.bias
        # upper = torch.max(max1, max2)  # + self.bias

        return torch.cat((middle, lower, upper), dim=1)


class Conv2dInterval(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, input_layer=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.importance = nn.Parameter(torch.zeros(self.weight.data.size()), requires_grad=True)
        self.eps = 0
        self.input_layer = input_layer

    def calc_eps(self, r):
        exp = self.importance.exp()
        # self.eps = r * exp / exp.sum()
        self.eps = r * exp / exp.sum(dim=-1).sum(dim=-1)[:, :, None, None]

    def rest_importance(self):
        # pass
        # w1 = torch.abs(1 / self.weight)
        # self.importance.data = w1 / w1.sum()
        # self.importance.data = w1 / w1.sum(dim=-1).sum(dim=-1)[:, :, None, None]
        self.importance.data = torch.zeros(self.weight.size()).cuda()
        # self.importance.data = torch.randn(self.weight.size()).cuda()

    def forward(self, x):
        if self.input_layer:
            x = torch.cat((x, x, x), dim=1)

        x_middle, x_lower, x_upper = split_activation(x)

        middle = super().forward(x_middle)

        w_lower_pos = (self.weight - self.eps).clamp(min=0)
        w_lower_neg = (self.weight - self.eps).clamp(max=0)
        w_upper_pos = (self.weight + self.eps).clamp(min=0)
        w_upper_neg = (self.weight + self.eps).clamp(max=0)

        lower = (f.conv2d(x_lower, w_lower_pos, None, self.stride,
                          self.padding, self.dilation, self.groups) +
                 f.conv2d(x_upper, w_lower_neg, None, self.stride,
                          self.padding, self.dilation, self.groups)) #+
                 # self.bias[None, :, None, None])

        upper = (f.conv2d(x_upper, w_upper_pos, None, self.stride,
                          self.padding, self.dilation, self.groups) +
                 f.conv2d(x_lower, w_upper_neg, None, self.stride,
                          self.padding, self.dilation, self.groups)) # +
                 # self.bias[None, :, None, None])

        return torch.cat((middle, lower, upper), dim=1)


if __name__ == '__main__':
    # li = LinearInterval(5, 3)
    # for n, p in li.named_parameters():
    #     print(f"name: {n}")
    #     print(n, p)

    pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    unpool = nn.MaxUnpool2d(2, stride=2)
    input = torch.tensor([[[[1., 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    input1 = torch.tensor([[[[1., 2, 3, 4], [5, 0, 7, 0], [9, 10, 11, 12], [13, 0, 15, 0]]]])

    print(input.size())
    print(input)
    output, ini = pool(input)
    print(f"output: {output.size()}, ini: {ini.size()}")
    print(output)
    print(ini)
    print()
    fini = torch.flatten(ini, 2)
    ft = torch.flatten(input1, 2)
    print(ft.size())
    print(ft[:, :, fini].view(output.size()))

    o = retrieve_elements_from_indices(input1, ini)
    print(o)

    # print(f"fini: {ini.size()}")
    # print(unpool(output, ini) == input)
    # print(input1[unpool(output, ini) == input])
    # mask = (unpool(output, ini) == input).long()
    # print(input1.gather(dim=2, index=mask))
