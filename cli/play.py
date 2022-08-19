from sys import argv
from utils import initialize, ignored, QLab

try:
	target = argv[-1]
	agent = initialize(target)
	with ignored(FileNotFoundError):
		agent.load(f"{QLab}/{target}/net")
	env = agent.env
except FileNotFoundError:
	exit("Agent does not exist")

with ignored(KeyboardInterrupt):
	for transition in agent.playoff():
		agent.env.render()
