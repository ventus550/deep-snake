import torch
from torch import nn
import torch.nn.functional as F
from snake import Environment
from QLearning import Agent, QNetwork, schedulers

width, height, vision = 20, 10, 1

class DeepNetwork(QNetwork):
	def __init__(self, input_size):
		super().__init__()
		self.connected1 = nn.Linear(input_size, 512)
		self.connected2 = nn.Linear(512, 256)
		self.connected3 = nn.Linear(256, 128)
		self.connected4 = nn.Linear(128, 4)

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = F.relu(self.connected1(x))
		x = F.relu(self.connected2(x))
		x = F.relu(self.connected3(x))
		x = F.relu(self.connected4(x))
		return x


env = Environment(shape = (width, height), vision = vision)
net = DeepNetwork( (2*vision + 1)**2 )
agent = Agent(net)
agent.train(env = env, live = False, scheduler = schedulers.linear, episodes=1000, plot=True)
agent.save()

