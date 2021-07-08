import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os

NET_NAME = "meinNetz.pt"

class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(10,10)
        self.lin2 = nn.Linear(10,10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num


if __name__ =="__main__":
    # print(netz)

    if os.path.isfile(NET_NAME):
        netz = torch.load(NET_NAME)
    else:
        netz = MeinNetz()



    for i in range(100):

        inputTensor = torch.Tensor([[1,0,0,0,1,0,0,0,1,1] for _ in range(10)])
        input = Variable(inputTensor) 
        # print(input)
        out = netz(input)
        # print(out)

        targetTensor = torch.Tensor([[0,1,1,1,0,1,1,1,0,0] for _ in range(10)])
        target = Variable(targetTensor)
        
        criterion = nn.MSELoss()
        loss = criterion(out, target)
        print(loss)


        netz.zero_grad()
        loss.backward()

        optimizer = optim.SGD(netz.parameters(), lr=0.1)
        optimizer.step()
    
    torch.save(netz, NET_NAME)

