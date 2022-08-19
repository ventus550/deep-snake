from sys import argv
from QLearning import schedulers
from utils import History, initialize, fetch, ignored, progress_bar, quickplot, QLab


def train(agent, path, scheduler = schedulers.linear, decay = 1.0, episodes = 1, update_frq = 10, live = False, plot = False):
	history = History()
	scheduler = scheduler(decay * episodes)
	with ignored(KeyboardInterrupt):
		for episode in progress_bar(range(episodes), disabled=live, desc="Training"):
			epsilon = next(scheduler)
			for transition in agent.playoff(epsilon = epsilon, train=True):

				# Store the transition in memory
				agent.memory.push(transition)

				# Perform one step of the optimization (on the policy network)
				agent.optimizer(gamma = agent.gamma)

				if live: 
					agent.env.render()

			history.store(
				epsilon = epsilon,
				score = agent.env.score,
				average = sum(history["score"])/(len(history["score"]) or 1)
			)

			# Update the target network, copying all weights and biases in DQN
			if episode % update_frq == 0:
				agent.target_net.copy_from(agent.policy_net)
			
			if plot and (episode % 100 == 0 or episode == episodes - 1):
				quickplot(history["epsilon"], ylabel = "Epsilon", path = f"{path}/epsilon")
				quickplot(history["score"], history["average"], legend=["Score", "Average"], path = f"{path}/score")
				agent.optimizer.plot_loss(f"{path}/loss")
				agent.optimizer.plot_loss_variance(f"{path}/variance")

try:
	target = argv[-1]
	agent = initialize(target)
	training = fetch(target, "agent")["training"]
	with ignored(FileNotFoundError):
		agent.load(f"{QLab}/{target}/net")
	env = agent.env
except FileNotFoundError:
	exit("Agent does not exist")

with ignored(KeyError):
	training["scheduler"] = schedulers.__dict__[training["scheduler"]]

train(agent, f"{QLab}/{target}", **training)
agent.save(f"{QLab}/{target}/net")
