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

    def __init__(self, hidden_size=8, learning_rate=0.1, epochs=1000):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W1 = torch.randn((2, hidden_size), requires_grad=False) * 0.1
        self.b1 = torch.randn((1, hidden_size), requires_grad=False) * 0.1
        self.W2 = torch.randn((hidden_size, 1), requires_grad=False) * 0.1
        self.b2 = torch.randn((1, 1), requires_grad=False) * 0.1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z1 = X @ self.W1 + self.b1
        a1 = torch.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        y_proba = torch.sigmoid(z2)
        self.z1 = z1
        self.a1 = a1
        self.z2 = z2
        self.y_proba = y_proba
        return y_proba

    def backward(self, X: torch.Tensor, y: torch.Tensor):
        dz2 = (self.y_proba - y) / X.size(0)
        self.dW2 = self.a1.T @ dz2
        self.db2 = dz2.sum(dim=0, keepdim=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - torch.tanh(self.z1) ** 2)
        self.dW1 = X.T @ dz1
        self.db1 = dz1.sum(dim=0, keepdim=True)


    def step(self):
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

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
    
