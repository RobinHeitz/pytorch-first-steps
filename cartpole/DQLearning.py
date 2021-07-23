import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
import math
import matplotlib.pyplot as plt
import gym
from PIL import Image

env = gym.make('CartPole-v0').unwrapped
if 'inline' in matplotlib.get_backend():
    from IPython import display

#interactive mode on
plt.ion()

#verallgemeinerung
# FloatTensor = torch.cuda.FloatTensor

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.pos] = (state, action, next_state, reward)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class Netz(nn.Module):
#Q learning: (action, state) -> expected reward

    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.norm1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.norm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.norm3 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(448,2)


    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))

resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])


width = 600


def cart_pos():
    env_width = env.x_threshold * 2
    return int(env.state[0] * width / env_width + width / 2.0)



def get_image():
    screen = env.render(mode='rgb_array').transpose( # brauchen in dimensionen CHW (Channel, height, width)
        (2,0,1)
    )
    screen = screen[:, 160:320]
    view = 320 #breite
    cart = cart_pos()
    if cart < view // 2:
        sliced = slice(view)
    elif cart > width - view // 2:
        sliced = slice(-1*view, None)
    else:
        sliced = slice(cart - view // 2, cart + view // 2)
    screen = screen[:, :, sliced]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen) # got tensor

    return resize(screen).unsqueeze(0).type(Tensor)


env.reset()
# plt.figure()
# plt.imshow(get_image().cpu().squeeze(0).permute(1,2,0).numpy(), interpolation='none')
# plt.show()


model = Netz()
# model = model.cuda()
optimizer = optim.RMSprop(model.parameters())
mem = Memory(16384)
done = 0

eps_end = 0.02
eps_start = 0.95
eps_steps = 250

def get_action(state):
    #exploration:
    #Wenn epsilon > treshold, dann wir das model verwendet, anderenfalls eine zufällige action ausgeführt.
    
    global done
    epsilon = random.random()
    threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * done / eps_steps)
    done = done + 1
    if epsilon > threshold:
        return model(Variable(state, volatile=True).type(Tensor)).data.max(1)[1].view(1,1)
    else:
        return LongTensor([random.randint(0,1)])












