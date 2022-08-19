import json
from sys import argv
from QLearning import schedulers
from utils import initialize, fetch, ignored, QLab

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

agent.train(**training)
agent.save(f"{QLab}/{target}/net")
