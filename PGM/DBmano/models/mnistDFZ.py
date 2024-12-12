import torch
import torch.nn as nn
import torch.nn.functional as F

"""
generator p(z)p(x|z)p(y|x, z), DFZ
"""

class DeconvLayer(nn.Module):
    def __init__(self, output_shape, filter_shape, activation, strides, name):
        super(DeconvLayer, self).__init__()
        self.output_shape = output_shape
        self.activation = activation
        self.strides = strides
        
        in_channels = filter_shape[2]
        out_channels = filter_shape[3]
        kernel_size = filter_shape[:2]
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=strides, padding=(kernel_size[0] // 2))

    def forward(self, x):
        x = self.deconv(x)
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'linear':
            return x
        elif self.activation == 'split':
            x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
            return torch.sigmoid(x1), x2


class MLP(nn.Module):
    def __init__(self, layers, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
            elif self.activation == 'linear':
                pass  # No activation
        return x


class ConvNet(nn.Module):
    def __init__(self, name, input_shape, filter_shapes, fc_layer_sizes, activation, last_activation):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = input_shape[0]
        for filter_shape in filter_shapes:
            self.layers.append(
                nn.Conv2d(in_channels, filter_shape[3], kernel_size=filter_shape[:2], stride=1, padding=filter_shape[0] // 2)
            )
            in_channels = filter_shape[3]
        self.fc_layers = MLP([int(in_channels * input_shape[1] * input_shape[2])] + fc_layer_sizes, activation)
        self.last_activation = last_activation

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        if self.last_activation == 'relu':
            x = F.relu(x)
        return x


def construct_filter_shapes(layer_channels, filter_width):
    return [(filter_width, filter_width, in_ch, out_ch) for in_ch, out_ch in zip(layer_channels[:-1], layer_channels[1:])]


class Generator(nn.Module):
    def __init__(self, input_shape, dimH, dimZ, dimY, n_channel, last_activation, name):
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
        self.decoder_input_shape = [(4, 4, n_channel), (7, 7, n_channel), (14, 14, n_channel), input_shape]
        fc_layers = [dimZ, dimH, int(torch.prod(torch.tensor(self.decoder_input_shape[0])))]
        self.mlp = MLP(fc_layers, activation='relu')

        self.conv_layers = nn.ModuleList()
        for i in range(len(self.decoder_input_shape) - 1):
            output_shape = self.decoder_input_shape[i + 1]
            input_shape = self.decoder_input_shape[i]
            strides = (
                int(output_shape[0] / input_shape[0]),
                int(output_shape[1] / input_shape[1])
            )
            activation = 'relu' if i < len(self.decoder_input_shape) - 2 else last_activation

            if activation in ['logistic_cdf', 'gaussian'] and i == len(self.decoder_input_shape) - 2:
                activation = 'split'
                output_shape = (output_shape[0], output_shape[1], output_shape[2] * 2)

            filter_shape = (filter_width, filter_width, input_shape[-1], output_shape[-1])
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


# Sampling from Gaussian

def sample_gaussian(mu, log_sig):
    std = torch.exp(log_sig)
    return mu + std * torch.randn_like(mu)
