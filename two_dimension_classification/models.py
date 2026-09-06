"""用 CPU 上的 torch.Tensor 手写模型。

参数保持 requires_grad=False，梯度公式和参数更新都自己实现。
不使用自动求导、loss.backward() 或 torch.optim。
"""

import torch


def binary_cross_entropy(y_proba: torch.Tensor, y: torch.Tensor):
    loss = -torch.mean(y * torch.log(y_proba + 1e-8) + (1 - y) * torch.log(1 - y_proba + 1e-8))
    return loss


class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = torch.zeros((2, 1), requires_grad=False)
        self.b = torch.zeros(1, requires_grad=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = X @ self.W + self.b
        y_proba = torch.sigmoid(logits)
        self.y_proba = y_proba
        return y_proba

    def backward(self, X: torch.Tensor, y: torch.Tensor):
        dz = (self.y_proba - y) / X.size(0)
        self.dW = X.T @ dz
        self.db = dz.sum(dim=0)

    def step(self):
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        losses = []
        for epoch in range(self.epochs):
            self.forward(X)
            loss = binary_cross_entropy(self.y_proba, y)
            losses.append(loss.item())
            self.backward(X, y)
            self.step()
        self.loss_list = losses

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    def judge(self, X: torch.Tensor) -> torch.Tensor:
        y_proba = self.predict_proba(X)
        judge_result = (y_proba >= 0.5).float()
        return judge_result


class ManualMLP:
    """单隐藏层神经网络：隐藏层用 tanh，输出层用 sigmoid。"""

    def __init__(self, hidden_size=8, learning_rate=0.1, epochs=1000):
        """待设置隐藏层大小、学习率、训练轮数，以及四个参数。

        W1: (2, hidden_size)，b1: (1, hidden_size)。
        W2: (hidden_size, 1)，b2: (1, 1)。
        """
        raise NotImplementedError("待实现：神经网络初始化")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """前向计算并保存中间结果，返回 (N, 1) 的概率。"""
        raise NotImplementedError("待实现：神经网络前向计算")

    def backward(self, X: torch.Tensor, y: torch.Tensor):
        """用链式法则手写并保存 dW1、db1、dW2、db2。"""
        raise NotImplementedError("待实现：神经网络梯度公式")

    def step(self):
        """根据保存的梯度和学习率，手动更新四个参数。"""
        raise NotImplementedError("待实现：神经网络参数更新")

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """组织前向计算、损失计算、梯度计算和更新，返回每轮损失列表。"""
        raise NotImplementedError("待实现：神经网络训练")

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """返回属于类别 1 的概率，形状为 (N, 1)。"""
        raise NotImplementedError("待实现：神经网络概率预测")

    def judge(self, X: torch.Tensor) -> torch.Tensor:
        """概率大于等于 0.5 时返回 1，否则返回 0，形状为 (N, 1)。"""
        raise NotImplementedError("待实现：神经网络类别判断")
