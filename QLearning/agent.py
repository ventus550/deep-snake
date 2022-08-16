import torch
from snake import Environment
from collections import defaultdict
from QLearning import QNetwork, Qptimizer, ReplayMemory, schedulers
from QLearning.utilities import quickplot, ignored, progress_bar

class Agent:
	def __init__(self, qnet : QNetwork, learning_rate = 0.01, gamma = 0.97, memory = 20000, criterion = torch.nn.MSELoss()):
		self.gamma = gamma
		self.memory = ReplayMemory(memory)

		self.policy_net = qnet
		self.target_net = qnet.clone()
		self.target_net.copy_from(self.policy_net)
		self.target_net.eval()

		# self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
		self.optimizer = Qptimizer(
			self.memory,
			self.optimizer,
			self.policy_net,
			self.target_net,
			criterion = criterion
		)

	def __call__(self, state, epsilon = 0):
		roll = torch.rand(1).item()
		if roll >= epsilon:
			with torch.no_grad():
				return self.policy_net(state.unsqueeze(0)).max(1)[1][0].item()
		return torch.randint(4, (1,)).item()

	def save(self, path = "./net"):
		self.policy_net.save(path)

	def load(self, path = "./net"):
		self.policy_net.load(path)
		self.target_net.copy_from(self.policy_net) # probably the right way to do this ...

	def train(self, env = Environment(), scheduler = schedulers.linear, decay = None, episodes = 1, update_frq = 10, live = False, plot = False):
		history = defaultdict(list)
		scheduler = scheduler(decay or episodes)
		with ignored(KeyboardInterrupt):
			for episode in progress_bar(range(episodes), disabled=live, desc="Training"):
				game_state = env.reset().get_state()
				epsilon = next(scheduler)
				while not env.terminal:
					if live: env.render()
					action = self(game_state, epsilon)
					old_state, action, reward, game_state = env.action(action)

					# Store the transition in memory
					self.memory.push(old_state, action, reward, game_state)

					# Perform one step of the optimization (on the policy network)
					self.optimizer(gamma = self.gamma)

				if live: 
					env.render()

				if plot:
					history["epsilon"].append(epsilon)
					history["score"].append(env.score)
					history["average"].append(sum(history["score"])/len(history["score"]))
					quickplot(history["epsilon"], ylabel = "Epsilon", path = "./epsilon")
					quickplot(history["score"], history["average"], legend=["Score", "Average"], path = "./score")
					self.optimizer.plot_loss("./loss")
					self.optimizer.plot_loss_variance("./variance")

				# Update the target network, copying all weights and biases in DQN
				if episode % update_frq == 0:
					self.target_net.copy_from(self.policy_net)