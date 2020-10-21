import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        net = []
        channels = [in_size] + (n_layers - 1) * [hidden_size] + [out_size]
        for i in range(n_layers):
            net.append(nn.Linear(channels[i], channels[i + 1]))
            net.append(nn.LayerNorm(channels[i + 1]))
            net.append(nn.ReLU())

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class RecurrentBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        hidden_size,
        n_outputs,
        n_layers,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs

        self.in_mlps = nn.ModuleList([
            MLP(1, hidden_size, hidden_size, n_layers) for i in range(n_inputs)
        ])

        self.out_mlp = MLP(
            hidden_size,
            hidden_size,
            hidden_size,
            n_layers,
        )

        self.gru = nn.GRU(
            n_inputs * hidden_size,
            hidden_size,
            batch_first=True,
        )

    def forward(self, inputs):
        x = torch.cat(
            [self.in_mlps[i](inputs[i]) for i in range(self.n_inputs)],
            -1,
        )
        x = self.gru(x)[0]

        return self.out_mlp(x)
