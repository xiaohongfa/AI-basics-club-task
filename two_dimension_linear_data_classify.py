import torch
import matplotlib.pyplot as plt
torch.manual_seed(26079100049)

def two_dimention_linear_data_produce(batch_size):
    x0 = torch.randn(batch_size,2) + torch.tensor([2.0,2.0])
    x1 = torch.randn(batch_size,2) + torch.tensor([-2.0,-2.0])
    x = torch.cat([x0,x1],dim=0)

    y0 = torch.zeros(batch_size,1)
    y1 = torch.ones(batch_size,1)
    y = torch.cat([y0,y1],dim=0)
    data = torch.cat([x,y],dim=1)
    return data[torch.randperm(data.size(0))]

if __name__ == "__main__":
    data = two_dimention_linear_data_produce(100)
    class_0 = data[data[:,2]==0]
    class_1 = data[data[:,2]==1]
    plt.scatter(class_0[:, 0], class_0[:, 1], c='red', marker='o')
    plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', marker='x')

    plt.xlabel("x1")
    plt.ylabel("x2")
        
    plt.show()
    