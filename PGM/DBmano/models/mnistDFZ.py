import torch
import torch.nn as nn
import torch.nn.functional as F
from .mnistGFZ import DeconvLayer, MLP, ConvNet, construct_filter_shapes

"""
DFZ model for MNIST dataset
"""

class Generator(nn.Module):
    def __init__(self, input_shape, dimH, dimZ, dimY, n_channel, last_activation, name='DFZ'):
        super(Generator, self).__init__()
        # p(y|z, x)
        layer_channels = [n_channel for _ in range(3)]
        filter_width = 5
        filter_shapes = construct_filter_shapes(layer_channels, filter_width)
        fc_layer_sizes = [dimH]
        self.gen_conv = ConvNet(name + '_pyzx_conv', input_shape, filter_shapes, fc_layer_sizes, activation='relu', last_activation='relu')

        fc_layers = [dimZ + dimH, dimH, dimY]
        self.pyzx_mlp = MLP(fc_layers, activation='linear')

        # p(x|z)
        self.decoder_input_shape = [
            (n_channel, 4, 4),
            (n_channel, 7, 7),
            (n_channel, 14, 14),
            input_shape  # (1,28,28)
        ]

        fc_layers = [dimZ, dimH, int(torch.prod(torch.tensor(self.decoder_input_shape[0])))]
        self.mlp = MLP(fc_layers, activation='relu')

        self.conv_layers = nn.ModuleList()
        for i in range(len(self.decoder_input_shape) - 1):
            output_shape = self.decoder_input_shape[i + 1]  # (C,H,W)
            input_shape = self.decoder_input_shape[i]        # (C,H,W)
            strides = (
                output_shape[1] // input_shape[1],  # stride en hauteur
                output_shape[2] // input_shape[2]   # stride en largeur
            )
            activation = 'relu' if i < len(self.decoder_input_shape) - 2 else last_activation

            if activation in ['logistic_cdf', 'gaussian'] and i == len(self.decoder_input_shape) - 2:
                activation = 'split'
                # doublez le nombre de canaux de sortie
                output_shape = (output_shape[0]*2, output_shape[1], output_shape[2])

            filter_shape = (filter_width, filter_width, input_shape[0], output_shape[0])
            self.conv_layers.append(DeconvLayer(output_shape, filter_shape, activation, strides, name))


    def pyzx_params(self, z, x):
        fea = self.gen_conv(x)
        out = torch.cat([fea, z], dim=1)
        return self.pyzx_mlp(out)

    def pxz_params(self, z):
        x = self.mlp(z)
        x = x.view(-1, *self.decoder_input_shape[0])
        for layer in self.conv_layers:
            x = layer(x)
        return x
