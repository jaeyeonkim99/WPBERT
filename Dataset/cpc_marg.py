import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
# from cpc_criterion import *


class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.05, is_causal=False, init=None, norm=False):
        super(ConvGLU, self).__init__()
        self.norm = norm
        if self.norm:
            self.instance_norm = nn.InstanceNorm1d(out_ch, affine=False)
        self.kernel = kernel 
        self.is_causal = is_causal
        self.dilation = dilation
        self.dropout = dropout
        if self.dropout == 0.0:
            pass
        else:
            self.drop = nn.Dropout(p=self.dropout)
        self.conv = nn.Conv1d(in_ch, out_ch*2, kernel_size=kernel, stride=1, dilation=dilation)
        if init == 'xavier':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        if self.is_causal:
            self.paddings = (self.dilation*(self.kernel-1), 0, 0, 0, 0, 0)
        else :
            self.paddings =  (self.dilation*(self.kernel-1)//2, self.dilation*(self.kernel-1)//2, 0, 0, 0, 0)
    def forward(self, input):
        if self.dropout == 0.0:
            pass
        else:
            x = self.drop(input)
        enc = F.pad(x, self.paddings, "constant", 0)
        enc = self.conv(enc)
        h1, h2 = enc.chunk(2,1)
        h = torch.sigmoid(h1) * h2
        if self.norm:
            h = self.instance_norm(h)
        h = h + input
        return h

class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.linear = nn.Conv1d(1024, dim, 1, 1)
        self.conv1 = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        self.norm1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm1d(dim)
        self.conv3 = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm1d(dim)
    def forward(self, x1):
        x1 = F.relu(self.linear(x1))
        x1 = F.relu(self.norm1(self.conv1(x1)))
        x2 = F.relu(self.norm2(self.conv2(x1)))
        x3 = F.relu(self.norm3(self.conv3(x2)))
        return x1, x2, x3

class ChannelNorm(nn.Module):
    def __init__(self,
                numFeatures,
                epsilon=1e-05,
                affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()
    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
    def forward(self, x):
        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x

class CPCEncoder(nn.Module):
    def __init__(self,
                sizeHidden=512,
                normMode="layerNorm"):
        super(CPCEncoder, self).__init__()
        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")
        if normMode == "instanceNorm":
            def normLayer(x): return nn.InstanceNorm1d(x, affine=True)
        elif normMode == "ID":
            normLayer = IDModule
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d
        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160
    def getDimOutput(self):
        return self.conv4.out_channels
    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x

class cLN(nn.Module):
    def __init__(self, input_dim=128, cond_dim=128):
        super(cLN, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(cond_dim, 128)
        self.fc2 = nn.Linear(128, input_dim*2)
    def forward(self, x, cond):
        x = self.layer_norm(x.transpose(1,2)).transpose(1,2)
        y = self.fc2(F.leaky_relu(self.fc1(cond)))
        mean, std = y.chunk(2,dim=-1)
        out = x * std.unsqueeze(2) + mean.unsqueeze(2)
        return out

class SingerEnc(nn.Module):
    def __init__(self, out_dim=128, mel_dim=80):
        super(SingerEnc, self).__init__()
        self.mel_dim = mel_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(self.mel_dim, self.out_dim, 5,1, padding=2)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv1d(self.out_dim, self.out_dim, 5,1, padding=2)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv1d(self.out_dim, self.out_dim, 3,1, padding=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.fc = nn.Linear(self.out_dim, self.out_dim)
        self.ln1 = nn.LayerNorm(self.out_dim)
        self.ln2 = nn.LayerNorm(self.out_dim)
        self.ln3 = nn.LayerNorm(self.out_dim)
    def forward(self, x):
        x = self.ln1(F.leaky_relu(self.conv1(x)).transpose(1,2)).transpose(1,2)
        x = self.ln2(F.leaky_relu(self.conv2(x)).transpose(1,2)).transpose(1,2)
        x = self.ln3(F.leaky_relu(self.conv3(x)).transpose(1,2)).transpose(1,2)
        x = torch.mean(x, dim=-1)               # [batch, 256]
        x = self.fc(x)                          # [batch, 256]
        # x = x.unsqueeze(-1)                     # [batch, 256, 1]
        return x

class Decoder(nn.Module):
    def __init__(self, in_dim=256, dim=256, out_dim=80, dropout=0.05):
        super(Decoder, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.conv_d1 = nn.Conv1d(in_dim+2, self.dim, 1, 1)
        torch.nn.init.xavier_uniform_(self.conv_d1.weight)
        self.conv_d2 = nn.Conv1d(self.dim, out_dim, 1, 1)
        torch.nn.init.xavier_uniform_(self.conv_d2.weight)
        torch.nn.init.constant_(self.conv_d2.bias, 0)
        self.drop = nn.Dropout(p=self.dropout)
        self.layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(ConvGLU(self.dim, self.dim, 3, 3**i, self.dropout, False))
            self.cond_layers.append(cLN(self.dim, 128))
        for i in range(4):
            self.layers.append(ConvGLU(self.dim, self.dim, 3, 3**i, self.dropout, False))
            self.cond_layers.append(cLN(self.dim, 128))
        for i in range(2):
            self.layers.append(ConvGLU(self.dim, self.dim, 1, 1, self.dropout, False))
            self.cond_layers.append(cLN(self.dim, 128))
    def forward(self, enc, spk=None):
        dec = F.leaky_relu(self.conv_d1(enc))
        for f, g in zip(self.layers, self.cond_layers):
            dec = f(dec)
            dec = g(dec, spk)
        dec = self.conv_d2(dec)
        return dec


class CPCAR(nn.Module):
    def __init__(self, dim=128):
        super(CPCAR, self).__init__()
        self.gru = nn.LSTM(dim, dim, num_layers=2, batch_first=True)
    def forward(self, x):
        try:
            self.gru.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.gru(x)
        return x


class BabyMind(nn.Module):
    def __init__(self, dim=256):
        super(BabyMind, self).__init__()
        self.enc = CPCEncoder(dim)
        self.spk_enc = SingerEnc()
        self.cpcar1 = CPCAR(dim)
        self.dec1 = Decoder(dim)
    def forward(self, x, label):
        # x [B, 160*200-1]
        
        encodedData = self.enc(x).permute(0,2,1)
        cFeature = self.cpcar1(encodedData)        

        return cFeature, encodedData, label

    def extract_features(self, source, get_encoded=False, norm_output=False):
        cpc_feature, encoded, _ = self.forward(source, None)
        if get_encoded:
            cpc_feature = encoded
        if norm_output:
            mean = cpc_feature.mean(dim=1, keepdim=True)
            var = cpc_feature.var(dim=1, keepdim=True)
            cpc_feature = (cpc_feature - mean) / torch.sqrt(var + 1e-08)
        return cpc_feature