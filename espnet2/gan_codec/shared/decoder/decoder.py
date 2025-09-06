from espnet2.gan_codec.shared.decoder.layers import WNConv1d, WNConvTranspose1d, Snake1d
import torch.nn as nn
import torch

class GLSTM(nn.Module):

    def __init__(self, in_features=None, out_features=None, hidden_size=896, groups=2):
        super().__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList(
            [nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList(
            [nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
        out = torch.chunk(out, self.groups, dim=-1)

        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1)).contiguous()

        out = out.transpose(1, 2).contiguous()

        return out


class ResidualUnit(nn.Module):

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2 * 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.ZeroPad1d((pad, 0)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=0),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y


class CausalDecoderBlock(nn.Module):

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=stride,
                stride=stride,
                padding=0,
                output_padding=0,  #stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9), 
        )

    def forward(self, x):
        return self.block(x)


class CausalDecoder(nn.Module):

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        groups: int = 1,
        lookahead_frame: int = 0,
        lstm_nums: int = 1,
    ):
        super().__init__()
        # [B, dim, T]
        # add mem layer here. keep shape.

        self.mem_layers = nn.ModuleList([GLSTM(groups=groups, hidden_size=input_channel) for _ in range(lstm_nums)])
        
        # Add first conv layer
        kernel_size = 2 * lookahead_frame + 1
        layers = [WNConv1d(input_channel, channels, kernel_size=kernel_size, padding=lookahead_frame)]

        
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers += [CausalDecoderBlock(input_dim, output_dim, stride)]
        layers += [
            Snake1d(output_dim),
            nn.ZeroPad1d((6, 0)),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        for mem_layer in self.mem_layers:
            x = mem_layer(x)
        x = self.model(x)
        return x
