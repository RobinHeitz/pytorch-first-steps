import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import os
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

NET_NAME = "catsanddogsnet.pt"


#Bild vorbearbeitung: Farben normalisieren & auf 256*256 zuschneiden

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize,
])

#TARGET: [isCat, isDog]

train_data_list = []
target_list = []
train_data = []
len_train_data = len(listdir('data/train/'))
files = listdir('data/train/')

for i in range(len_train_data):
    f = random.choice(files)
    files.remove(f)

    img = Image.open("data/train/"+f)
    img_tensor = transform(img) #(3,256,256)
    train_data_list.append(img_tensor)
    
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]

    target_list.append(target)

    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        print('Loaded batch ', len(train_data), 'of ', int(len_train_data / 64))
        print('Percentage Done: ', int(100 * len(train_data) / (len_train_data / 64)), ' %')
        break
               
        

print(train_data)


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3,6, kernel_size=5)
        self.conv2 = nn.Conv2d(6,12, kernel_size=5)
        self.conv3 = nn.Conv2d(12,18, kernel_size=5)
        self.conv4 = nn.Conv2d(18,24, kernel_size=5)
        
        self.fc1 = nn.Linear(3456,1000)
        self.fc2 = nn.Linear(1000,2)

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
       
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # print(x.size())
        # exit()

        x = x.view(-1,3456)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)
        


if os.path.isfile(NET_NAME):
    model = torch.load(NET_NAME)
else:
    model = Netz()

# model.cuda()


optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        # data = data.cuda()
        # target = torch.Tensor(target).cuda()
        target = torch.Tensor(target)
        data = Variable(data)
        target = Variable(target)

        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        # print('Train Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_id * len(data), len(train_data), 100. * batch_id / len(train_data), loss.item()))
        # batch_id = batch_id + 1
    torch.save(model, NET_NAME)


def test():
    model.eval()
    files = listdir('data/test/')
    f = random.choice(files)
    img = Image.open('data/test/'+f)
    img_eval_tensor = transform(img)
    img_eval_tensor.unsqueeze_(0)
    # data = Variable(img_eval_tensor.cuda())
    data = Variable(img_eval_tensor)
    out = model(data)

    label = out.data.max(1, keepdim=True)[1].item()
    if label == 0:
        print('Cat')
    else:
        print('Dog')

    img.show()
    try:

        x = input('')
    except KeyboardInterrupt:
        return



for epoch in range(1,30):
    train(epoch)
    # test()
    