"""生成二维二分类的线性和非线性数据，并划分训练集和测试集。"""
import torch


def two_dimention_linear_data_generate(batch_size: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    std = 1.0  
    x0 = torch.randn(batch_size, 2) * std + torch.tensor([2.0, 2.0])
    x1 = torch.randn(batch_size, 2) * std + torch.tensor([-2.0, -2.0])
    y0 = torch.zeros(batch_size, 1)
    y1 = torch.ones(batch_size, 1)
    data = torch.cat([torch.cat([x0, y0], dim=1),
                      torch.cat([x1, y1], dim=1)], dim=0)
    return data[torch.randperm(data.size(0))] 


def two_dimention_nonlinear_data_generate(batch_size: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    theta1 = torch.randn(batch_size) * 2 * torch.pi
    theta2 = torch.randn(batch_size) * 2 * torch.pi
    r1 = 1.0 + 0.1 * torch.rand(batch_size)
    r2 = 3.0 + 0.1 * torch.rand(batch_size)
    x1 = torch.stack([
        r1 * torch.cos(theta1),
        r1 * torch.sin(theta1)
    ], dim=1)
    x2 = torch.stack([
        r2 * torch.cos(theta2),
        r2 * torch.sin(theta2)
    ], dim=1)
    y1 = torch.zeros(batch_size, 1)
    y2 = torch.ones(batch_size, 1)
    data1 = torch.cat([x1, y1], dim=1)
    data2 = torch.cat([x2, y2], dim=1)
    data = torch.cat([data1, data2], dim=0)

    return data[torch.randperm(data.size(0))]

    


def split_data(data: torch.Tensor, test_ratio: float = 0.2):
    n_test = int(data.size(0) * test_ratio)
    X_test = data[:n_test, :2]
    Y_test = data[:n_test, 2:3]
    X_train = data[n_test:, :2]
    Y_train = data[n_test:, 2:3]
    return X_train, X_test, Y_train, Y_test
