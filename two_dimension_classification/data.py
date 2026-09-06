"""生成二维二分类的线性和非线性数据，并划分训练集和测试集。"""
import torch
torch.manual_seed(26079100049)

def two_dimention_linear_data_generate(batch_size: int, seed: int) -> torch.Tensor:
    x0 = torch.rand(batch_size, 2) + torch.tensor([2.0, 2.0])  
    x1 = torch.rand(batch_size, 2) + torch.tensor([-2.0, -2.0])
    y0 = torch.zeros(batch_size, 1)
    y1 = torch.ones(batch_size, 1)
    data = torch.cat([torch.cat([x0, y0], dim=1),
                      torch.cat([x1, y1], dim=1)], dim=0)
    return data[torch.randperm(data.size(0))]  # 打乱数据


def two_dimention_nonlinear_data_generate(batch_size: int, seed: int) -> torch.Tensor:
    """生成半径约为 1 和 3 的两个带噪声圆环。

    batch_size 表示每个类别的点数。
    返回打乱后的 (2 * batch_size, 3) Tensor，每行是 [x1, x2, label]。
    """
    raise NotImplementedError("待实现：非线性数据生成")


def split_data(data: torch.Tensor, test_ratio: float = 0.2):
    n_test = int(data.size(0) * test_ratio)
    X_test = data[:n_test, :2]
    Y_test = data[:n_test, 2:3]
    X_train = data[n_test:, :2]
    Y_train = data[n_test:, 2:3]
    return X_train, X_test, Y_train, Y_test
