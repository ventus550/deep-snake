import torch
from snake import Environment
from QLearning import QNetwork, Qptimizer, ReplayMemory, schedulers
from QLearning.utils import History, quickplot, ignored, progress_bar

class Agent:
	"""
	Entity capable of interacting with the game environment.

	Attributes:
		vision		-- perceived game state range

		gamma		-- uncertainty discount for future actions (should be close but not greater than 1.0)

		optimizer	-- used optimization algorithm for the gradient learning method 
	"""

	def __init__(self, qnet : QNetwork, env : Environment, learning_rate = 0.01, gamma = 0.97, memory = 20000, vision = 1, criterion = torch.nn.MSELoss()):
		self.env = env
		self.gamma = gamma
		self.vision = vision
		self.memory = ReplayMemory(memory)

		self.policy_net = qnet
		self.target_net = qnet.clone()
		self.target_net.copy_from(self.policy_net)
		self.target_net.eval()

		# self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate) # Adam is way better lol
		self.optimizer = Qptimizer(
			self.memory,
			self.optimizer,
			self.policy_net,
			self.target_net,
			criterion = criterion
		)

	def __call__(self, state, epsilon = 0):
		"Choose an action following epsilon-greedy strategy"
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

	def train_mode(self):
		self.policy_net.train()
		self.target_net.train()
	
	def eval_mode(self):
		self.policy_net.eval()
		self.target_net.eval()

	def playoff(self, epsilon = 0):
		game_state = self.env.reset().get_state(self.vision)
		while not self.env.terminal:
			action = self(game_state, epsilon)
			old_state, action, reward, game_state = self.env.action(action)
			yield old_state, action, reward, game_state

	def train(self, scheduler = schedulers.linear, decay = 1.0, episodes = 1, update_frq = 10, live = False, plot = False):
		history = History()
		scheduler = scheduler(decay * episodes)
		with ignored(KeyboardInterrupt):
			for episode in progress_bar(range(episodes), disabled=live, desc="Training"):
				game_state = self.env.reset().get_state(self.vision)
				epsilon = next(scheduler)
				while not self.env.terminal:
					if live:
						self.env.render()
					action = self(game_state, epsilon)
					transition = self.env.action(action)

					# Store the transition in memory
					self.memory.push(transition)

					# Perform one step of the optimization (on the policy network)
					self.optimizer(gamma = self.gamma)

				if live: 
					self.env.render()

				history.store(
					epsilon = epsilon,
					score = self.env.score,
					average = sum(history["score"])/(len(history["score"]) or 1)
				)

				# Update the target network, copying all weights and biases in DQN
				if episode % update_frq == 0:
					self.target_net.copy_from(self.policy_net)
					if plot:
						quickplot(history["epsilon"], ylabel = "Epsilon", path = "./epsilon")
						quickplot(history["score"], history["average"], legend=["Score", "Average"], path = "./score")
						self.optimizer.plot_loss("./loss")
						self.optimizer.plot_loss_variance("./variance")