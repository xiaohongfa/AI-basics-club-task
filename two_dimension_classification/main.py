"""依次组织线性与非线性两个实验。"""

from data import (
    two_dimention_linear_data_generate,
    two_dimention_nonlinear_data_generate,
    split_data,
)
from models import LogisticRegression, ManualMLP
from visualize import plot_decision_boundary, plot_loss_curve


def run_linear_experiment():
    """线性数据 + 逻辑回归。"""
    # TODO: 1. 生成线性数据，再划分训练集和测试集。
    # TODO: 2. 创建 LogisticRegression，调用 fit 并保存损失列表。
    # TODO: 3. 调用 judge，与 y_test 比较，计算测试准确率。
    # TODO: 4. 绘制决策边界和损失曲线，将图片保存到 results/。
    data = two_dimention_linear_data_generate(batch_size=1000, seed=26079100049)
    x_train, x_test, y_train, y_test = split_data(data, test_ratio=0.2)
    model = LogisticRegression()
    losses = model.fit(x_train, y_train)
    accuracy = (model.judge(x_test) == y_test).float().mean().item()
    print(f"测试准确率：{accuracy:.2%}喵")
    plot_decision_boundary(model, x_test, y_test)
    plot_loss_curve(model.loss_list)


def run_nonlinear_experiment():
    """圆环数据 + 单隐藏层神经网络。"""
    # TODO: 1. 生成非线性数据，再划分训练集和测试集。
    # TODO: 2. 创建 ManualMLP，调用 fit 并保存损失列表。
    # TODO: 3. 调用 judge，与 y_test 比较，计算测试准确率。
    # TODO: 4. 绘制决策边界和损失曲线，将图片保存到 results/。
    raise NotImplementedError("待实现：非线性实验流程")


def main():
    run_linear_experiment()
    #run_nonlinear_experiment()


if __name__ == "__main__":
    main()
