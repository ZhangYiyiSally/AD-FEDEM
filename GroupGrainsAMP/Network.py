import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ResidualBlock(nn.Module):  # 残差块
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.activate = nn.Tanh()  # 激活函数

         # 初始化参数
        torch.nn.init.normal_(self.fc1.bias, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc2.bias, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=0.1)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activate(out)

        out = self.fc2(out)
        out += residual  # 添加残差连接
        out = self.activate(out)  # 激活函数
        return out

class ResNet(nn.Module):  # 残差神经网络
    def __init__(self, input_size: int, hidden_size: int, output_size: int, depth: int, data_num: int = 1, latent_dim: int = 32):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth
        self.data_num = data_num
        self.latent_dim = latent_dim

        self.layers = nn.ModuleList()  # 使用 ModuleList 来存储所有的层
        self.layers.append(nn.Linear(input_size + latent_dim, hidden_size))  # 添加输入层

        # 初始化输入层的参数
        nn.init.normal_(self.layers[0].bias, mean=0, std=0.001)
        nn.init.normal_(self.layers[0].weight, mean=0, std=0.001)

        for i in range(depth - 1):  # 添加残差块
            self.layers.append(ResidualBlock(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))  # 添加输出层

        # 初始化输出层的参数
        nn.init.normal_(self.layers[-1].bias, mean=0, std=0.001)
        nn.init.normal_(self.layers[-1].weight, mean=0, std=0.001)
        
        self.layers.append(nn.Sigmoid()) # 限制网络输出范围(0, 1)

        self.latent_vectors = nn.Parameter(torch.FloatTensor(data_num, latent_dim)) # 隐变量向量
        nn.init.xavier_normal(self.latent_vectors)

    def denormalize(self, y, ymin=-1, ymax=1): # 反归一化，把输出范围线性缩放至(ymin, ymax)
        return y * (ymax - ymin) + ymin
    
    def x_concat(self, x, idx): # 将隐变量向量与输入数据拼接
        latent=self.latent_vectors[idx]
        original_shape = x.shape
        flattened = x.view(-1, original_shape[-1])  # (batch_size*seq_len, 3)
            
        # 扩展隐变量
        latent_expanded = latent.unsqueeze(0).expand(flattened.size(0), -1)  # (batch_size*seq_len, latent_dim)
            
        # 拼接后恢复原始维度
        combined = torch.cat([flattened, latent_expanded], dim=1)  # (batch_size*seq_len, 3+latent_dim)
        x = combined.view(*original_shape[:-1], -1)  # (..., seq_len, 3+latent_dim)

        return x

    def forward(self, x, data_idx):
        x= self.x_concat(x, data_idx)  # 拼接隐变量向量
        for layer in self.layers:
            x = layer(x)
        x = self.denormalize(x)
        return x


if __name__ == '__main__':
    # 创建神经网络
    model = ResNet(input_size=3, hidden_size=128, output_size=3, depth=3, data_num=3, latent_dim=32)
    data1=torch.randn(10, 3)  # 随机输入数据
    data2=torch.randn(20, 5, 3)  # 随机输入数据
    data3=torch.randn(15, 4, 2, 3)  # 随机输入数据
    data_list = [data1, data2, data3]  # 假设有三个数据集
    data_dict={}
    for i in range(3):
        data_dict[i]=data_list[i]
    data_num=3  # 数据索引
    mixed_idx = [0, 1, 2]
    for i in range(data_num):
        data = data_dict[i]
        output=model(data, i)  # 假设数据索引为0
