"""依次组织线性与非线性两个实验。"""

from data import (
    two_dimention_linear_data_generate,
    two_dimention_nonlinear_data_generate,
    split_data,
)
from models import LogisticRegression, ManualMLP
from visualize import plot_decision_boundary, plot_loss_curve


def run_linear_experiment():
    data = two_dimention_linear_data_generate(batch_size=1000, seed=26079100049)
    x_train, x_test, y_train, y_test = split_data(data, test_ratio=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    accuracy = (model.judge(x_test) == y_test).float().mean().item()
    print(f"测试准确率：{accuracy:.2%}喵")
    plot_decision_boundary(model, x_test, y_test)
    plot_loss_curve(model.loss_list)


def run_nonlinear_experiment():
    data = two_dimention_nonlinear_data_generate(batch_size=1000, seed=26079100049)
    x_train, x_test, y_train, y_test = split_data(data, test_ratio=0.2)
    model = ManualMLP()
    model.fit(x_train, y_train)
    accuracy = (model.judge(x_test) == y_test).float().mean().item()
    print(f"测试准确率：{accuracy:.2%}喵")
    plot_decision_boundary(model, x_test, y_test)
    plot_loss_curve(model.loss_list)



def main():
    run_linear_experiment()
    run_nonlinear_experiment()


if __name__ == "__main__":
    main()
