"""绘制实验结果，并将图片保存到 results/。"""


def plot_decision_boundary(predict_fn, X, y, save_path):
    """绘制决策边界，即模型划分两个类别的分界线。

    predict_fn 传入模型的 judge 方法；X 为 (N, 2)，y 为 (N, 1)。
    save_path 是图片保存路径，两个实验应使用不同文件名。
    """
    raise NotImplementedError("待实现：决策边界绘图")


def plot_loss_curve(loss_history, save_path):
    """根据 fit 返回的每轮损失列表绘图，并保存到 save_path。"""
    raise NotImplementedError("待实现：损失曲线绘图")
