"""绘制实验结果，并将图片保存到 results/。"""
import matplotlib.pyplot as plt
import torch

def plot_decision_boundary(model, X, y):
    # 铺网格
    xs = torch.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    ys = torch.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # 模型预测
    p = model.predict_proba(points).reshape(xx.shape)

    # 画真实样本和概率为 0.5 的边界
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="coolwarm")
    plt.contour(xx, yy, p, levels=[0.5], colors="black")
    plt.show()


def plot_loss_curve(loss_history):
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
