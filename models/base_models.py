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
