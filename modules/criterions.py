import torch

class BCEauto(torch.nn.BCEWithLogitsLoss):
    """
    BCE with logits loss + automatically convert the target from class label to one-hot vector
    """
    def forward(self, x, y):
        assert x.ndimension() == 2, 'Input size must be 2D'
        assert y.numel() == x.size(0), 'The size of input and target doesnt match. Number of input:' + str(x.size(0)) + ' Number of target:' + str(y.numel())
        y_onehot = x.clone().zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)

        return super(BCEauto, self).forward(x, y_onehot)