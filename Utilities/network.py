import torch
from torch import nn
import torch.nn.functional as F


class net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, output_activiation = None):
        """
        input_size: int, numero di neuroni in input
        hidden_sizes: list[int], dimensioni dei layer nascosti (può anche essere vuota)
        output_size: int, numero di neuroni in output
        """
        super(net, self).__init__()
        self.output_activation = output_activiation

        # Crea una lista vuota di layer
        layers = []

        # Primo layer: input -> primo hidden
        if hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.ReLU())

            # Hidden -> Hidden
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())

            # Ultimo hidden -> output
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        else:
            # Se non ci sono hidden, input -> output diretto
            layers.append(nn.Linear(input_size, output_size))

        # Salva come Sequential
        self.rete = nn.Sequential(*layers)

    def forward(self, x):
        x = self.rete(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x


if __name__ == "__main__":
    model = net(4, [8, 8], 2)
    x = model(torch.randn(1, 4))
    print(x)