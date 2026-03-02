import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.activate = nn.Tanh()

        torch.nn.init.normal_(self.fc1.bias, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc2.bias, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=0.1)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activate(out)

        out = self.fc2(out)
        out += residual
        out = self.activate(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, depth: int, data_num: int = 1, latent_dim: int = 32):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.data_num = data_num
        self.latent_dim = latent_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size + latent_dim, hidden_size))

        nn.init.normal_(self.layers[0].bias, mean=0, std=0.001)
        nn.init.normal_(self.layers[0].weight, mean=0, std=0.001)

        for _ in range(depth - 1):
            self.layers.append(ResidualBlock(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))

        nn.init.normal_(self.layers[-1].bias, mean=0, std=0.001)
        nn.init.normal_(self.layers[-1].weight, mean=0, std=0.001)

        self.layers.append(nn.Sigmoid())

        self.latent_vectors = nn.Parameter(torch.FloatTensor(data_num, latent_dim))
        nn.init.xavier_normal_(self.latent_vectors)

    def denormalize(self, y, ymin=-1, ymax=1):
        return y * (ymax - ymin) + ymin

    def x_concat(self, x, idx):
        if torch.is_tensor(idx):
            idx = idx.to(self.latent_vectors.device).long()
            latent = self.latent_vectors[idx]
            if latent.dim() == 1:
                latent_shape = (1,) * (x.dim() - 1) + (self.latent_dim,)
                latent_expanded = latent.view(*latent_shape).expand(*x.shape[:-1], self.latent_dim)
            else:
                latent_shape = (latent.shape[0],) + (1,) * (x.dim() - 2) + (self.latent_dim,)
                latent_expanded = latent.view(*latent_shape).expand(*x.shape[:-1], self.latent_dim)
        else:
            latent = self.latent_vectors[int(idx)]
            latent_shape = (1,) * (x.dim() - 1) + (self.latent_dim,)
            latent_expanded = latent.view(*latent_shape).expand(*x.shape[:-1], self.latent_dim)

        return torch.cat([x, latent_expanded], dim=-1)

    def forward(self, x, data_idx):
        x = self.x_concat(x, data_idx)
        for layer in self.layers:
            x = layer(x)
        x = self.denormalize(x)
        return x


if __name__ == '__main__':
    model = ResNet(input_size=3, hidden_size=128, output_size=3, depth=3, data_num=3, latent_dim=32)
    data1 = torch.randn(10, 3)
    data2 = torch.randn(20, 5, 3)
    data3 = torch.randn(15, 4, 2, 3)
    data_list = [data1, data2, data3]
    data_dict = {}
    for i in range(3):
        data_dict[i] = data_list[i]
    data_num = 3
    for i in range(data_num):
        data = data_dict[i]
        output = model(data, i)
