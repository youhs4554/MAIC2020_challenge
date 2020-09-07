from torch.nn import functional as F
from torch import nn
import torch.nn as nn
import torch


__all__ = ["ConvNd"]


class ConvNd(nn.Sequential):
    __nonlin_dict = {
        "relu": "ReLU",
        "lrelu": "LeakyReLU",
    }

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros",
                 nonlinearity='relu', dimension=1):

        super().__init__()

        assert 1 <= dimension <= 3, f"not supported dimension of {dimension}, should be in [1...3]"

        conv_name = "Conv{}d".format(dimension)
        bn_name = "BatchNorm{}d".format(dimension)

        conv_func = getattr(nn, conv_name)
        bn_func = getattr(nn, bn_name)
        nonlin_func = getattr(nn, self.__nonlin_dict[nonlinearity])

        self.add_module("conv", conv_func(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode))
        self.add_module("bn", bn_func(out_channels))
        self.add_module(nonlinearity, nonlin_func())


class NoneLocalBlockNd(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        sub_sample=True,
        bn_layer=True,
    ):
        super(NoneLocalBlockNd, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size, dim = x.size()[:2]

        # g_x  : (b,inter,t,h,w) -> (b, inter, t*h*w) -> (b, t*h*w, inter)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta : (b,inter,t,h,w) -> (b, inter, t*h*w) -> (b, t*h*w, inter)
        theta = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta = theta.permute(0, 2, 1)

        # phi : (b,inter,t,h,w) -> (b, inter, t*h*w)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta, phi)
        f_div_C = F.softmax(f, dim=-1)  # (b, t*h*w, t*h*w)

        y = torch.matmul(f_div_C, g_x)  # (b, t*h*w, inter)
        y = y.permute(0, 2, 1).contiguous()  # (b, inter, t*h*w)
        y = y.view(
            batch_size, self.inter_channels, *x.size()[2:]
        )  # (b, inter, t, h, w)
        W_y = self.W(y)  # (b, c_in, t, h, w)
        z = W_y + x

        return z


class StocasticPoolNd(nn.Module):
    def __init__(self, kernel_size, stride=None, dimension=1):
        super().__init__()

        assert 1 <= dimension <= 3, f"not supported dimension of {dimension}, should be in [1...3]"

        if isinstance(kernel_size, int):
            kernel_size = tuple([kernel_size]*dimension)
        if isinstance(stride, int):
            stride = tuple([stride]*dimension)

        if stride is not None:
            stride = kernel_size


    def forward(self, x):
        if self.train:
            pass
        else:
            pass
        # import torch.nn.functional as F
        # ...
        # F.conv1d(x, kernel)
        
        return x
