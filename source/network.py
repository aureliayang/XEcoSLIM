import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

class To_Latent(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        return out

class Network2(nn.Module):
    def __init__(self, dim, num_encoder_layers, num_decoder_layers, latent_dim):
        super(Network2, self).__init__()
        w0 = 50.
        w1 = 1.0
        self.dim = dim
        pos_latent_dim = int(latent_dim / 4 * 3)
        fc_latent_dim = int(latent_dim / 4)
        self.pos_encoder = nn.ModuleList()
        for i in range(num_encoder_layers):
            if i == 0:
                # print(i, dim, int(pos_latent_dim // 2**(num_encoder_layers - i - 1)))
                first_layer = To_Latent(dim, int(pos_latent_dim // 2**(num_encoder_layers - i - 1)), w0=w0, use_bias = True, is_first=True)
                self.pos_encoder.append(first_layer)
                self.pos_encoder.append(Sine(w0))
                #self.pos_encoder.append(nn.ReLU())
            else:
                # print(i, int(pos_latent_dim // 2**(num_encoder_layers - i)), int(pos_latent_dim // 2**(num_encoder_layers - i-1)))
                layer = To_Latent(int(pos_latent_dim // 2**(num_encoder_layers - i)), int(pos_latent_dim // 2**(num_encoder_layers - i - 1)), w0=w1, use_bias = True, is_first=False)
                self.pos_encoder.append(layer)
                self.pos_encoder.append(Sine(w1))
                #self.pos_encoder.append(nn.ReLU())
        self.pos_encoder = self.pos_encoder[:-1]
        
        self.activation = Sine(w1)
        
        self.fc_encoder = nn.ModuleList()
        for i in range(num_encoder_layers):
            if i == 0:
                first_layer = To_Latent(1, int(fc_latent_dim // 2**(num_encoder_layers - i - 1)), w0=w0, use_bias = True, is_first=True)
                self.fc_encoder.append(first_layer)
                self.fc_encoder.append(Sine(w0))
                #self.fc_encoder.append(nn.ReLU())
            else:
                layer = To_Latent(int(fc_latent_dim // 2**(num_encoder_layers - i)), int(fc_latent_dim // 2**(num_encoder_layers - i - 1)), w0=w1, use_bias = True, is_first=False)
                self.fc_encoder.append(layer)
                self.fc_encoder.append(Sine(w1))
                #self.fc_encoder.append(nn.ReLU())
        self.fc_encoder = self.fc_encoder[:-1]

        self.decoder = nn.ModuleList()
        for i in range(num_decoder_layers):
            if i == 0:    
                first_layer = To_Latent(latent_dim, latent_dim, w0=w0, use_bias = True, is_first=True)
                self.decoder.append(first_layer)
                self.decoder.append(Sine(w0))
                #self.decoder.append(nn.ReLU())
            elif i == num_decoder_layers - 1:
                # print(i, int(latent_dim // 2**(num_decoder_layers - 2)), dim )
                last_layer = To_Latent(int(latent_dim // 2**(num_decoder_layers - 2)), dim, w0=w1, use_bias = True, is_first=False)
                self.decoder.append(last_layer)
                self.decoder.append(Sine(w1))
                #self.decoder.append(nn.ReLU())
            else:
                # print(i, int(latent_dim // 2**(i - 1)), int(latent_dim // 2**(i)))
                layer = To_Latent(int(latent_dim // 2**(i-1)), int(latent_dim // 2**(i)), w0=w1, use_bias = True, is_first=False)
                self.decoder.append(layer)
                self.decoder.append(Sine(w1))
                #self.decoder.append(nn.ReLU())

    def forward(self, start, t):
        # B_pos = 1 * np.random.normal(size=(start.size(0), 2))
       
        # start = np.concatenate([np.sin(start @ B_pos), np.cos(start @ B_pos)], axis = -1)
        # print(start.size())
        for i, layer in enumerate(self.pos_encoder):
            start = layer(start)
        start = self.activation(start)
        for i, layer in enumerate(self.fc_encoder):
            t = layer(t)
        t = self.activation(t)
        x = torch.cat((start, t), axis=1)
        # print("done", x.size())
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x



if __name__ == "__main__":
    model = Network2(2, 4, 4, 512)
    # print(model)
    p = torch.rand(2, 2)
    t = torch.rand(2, 1)
    x = model(p, t)
    print("x.shape", x.shape)
