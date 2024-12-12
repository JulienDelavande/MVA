import torch
import torch.nn as nn
import torch.nn.functional as F

"""
generative model p(z)p(y|z)p(x|y, z), GFZ
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


class Generator(nn.Module):
    def __init__(self, input_shape, dimH, dimZ, dimY, n_channel, last_activation, name):
        super(Generator, self).__init__()
        # p(y|z)
        pyz_layers = [dimZ, dimH, dimY]
        self.pyz_mlp = MLP(pyz_layers, activation='linear')

        # p(x|z, y)
        self.decoder_input_shape = [(4, 4, n_channel), (7, 7, n_channel), (14, 14, n_channel), input_shape]
        fc_layers = [dimZ + dimY, dimH, int(torch.prod(torch.tensor(self.decoder_input_shape[0])))]
        self.mlp = MLP(fc_layers, activation='relu')

        self.conv_layers = nn.ModuleList()
        filter_width = 5

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

    def pyz_params(self, z):
        return self.pyz_mlp(z)

    def pxzy_params(self, z, y):
        x = torch.cat([z, y], dim=1)
        x = self.mlp(x)
        x = x.view(-1, *self.decoder_input_shape[0])
        for layer in self.conv_layers:
            x = layer(x)
        return x


# Sampling from Gaussian

def sample_gaussian(mu, log_sigma):
    std = torch.exp(log_sigma)
    return mu + std * torch.randn_like(mu)

# Example usage
# generator = Generator(input_shape=(28, 28, 1), dimH=128, dimZ=10, dimY=10, n_channel=16, last_activation='sigmoid', name='generator')
