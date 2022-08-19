import torch
from torch import nn
import torch.nn.functional as F
from QLearning import QNetwork


class ConnectedPure(QNetwork):
	def __init__(self, vision):
		super().__init__()
		self.connected1 = nn.Linear((2*vision + 1)**2, 512)
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