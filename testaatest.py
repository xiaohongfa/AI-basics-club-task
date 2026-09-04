import torch
import torch.nn as nn

model = nn.Linear(2, 1)

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

output = model(x)

print("weight =", model.weight)
print("bias =", model.bias)
print("output =", output)