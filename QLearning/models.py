import torch
from torch import nn
import torch.nn.functional as F
from QLearning import QNetwork
from QLearning.utils import flattened


class Simple(QNetwork):
	def __init__(self, vision):
		super().__init__()
		self.connected1 = nn.Linear(flattened(vision), 2048)
		self.connected2 = nn.Linear(2048, 4)

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = F.relu(self.connected1(x))
		x = F.relu(self.connected2(x))
		return x

class FullyConnected(QNetwork):
	def __init__(self, vision):
		super().__init__()
		self.connected1 = nn.Linear(flattened(vision), 512)
		self.connected2 = nn.Linear(512, 256)
		self.connected3 = nn.Linear(256, 128)
		self.connected4 = nn.Linear(128, 4)
	
	def transform(self, x):
		return F.relu(x)

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = self.transform(self.connected1(x))
		x = self.transform(self.connected2(x))
		x = self.transform(self.connected3(x))
		x = F.relu(self.connected4(x))
		return x

class FullyConnectedDropout(FullyConnected):
	def __init__(self, vision):
		super().__init__(vision)
		self.dropout = nn.Dropout(0.90)
	
	def transform(self, x):
		return self.dropout(super().transform(x))

class FullyConnectedNormalized(QNetwork):
	def __init__(self, vision):
		super().__init__()
		self.connected1 = nn.Linear(flattened(vision), 512)
		self.norm1 = nn.LayerNorm(512)
		self.connected2 = nn.Linear(512, 256)
		self.norm2 = nn.LayerNorm(256)
		self.connected3 = nn.Linear(256, 128)
		self.norm3 = nn.LayerNorm(128)
		self.connected4 = nn.Linear(128, 4)
		self.norm4 = nn.LayerNorm(4)

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = self.norm1(F.relu(self.connected1(x)))
		x = self.norm2(F.relu(self.connected2(x)))
		x = self.norm3(F.relu(self.connected3(x)))
		x = self.norm4(F.relu(self.connected4(x)))
		return x

class Dueling(QNetwork):
	def __init__(self, vision):
		super().__init__()
		
		self.feature_layer = nn.Sequential(
			nn.Linear(flattened(vision), 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU()
		)

		self.value_stream = nn.Sequential(
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

		self.advantage_stream = nn.Sequential(
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 4)
		)

	def forward(self, x):
		x = torch.flatten(x, 1)
		features = self.feature_layer(x)
		values = self.value_stream(features)
		advantages = self.advantage_stream(features)
		qvals = values + (advantages - advantages.mean())
		return qvals

class Convolutions(QNetwork):
	def __init__(self, vision):
		super().__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
		)

		self.fc = nn.Sequential(
			nn.Linear(flattened(vision) * 16, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, 4),
			nn.ReLU()
		)

	def forward(self, x):
		x = x.unsqueeze(1)
		x = self.conv(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x