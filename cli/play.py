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
	game_state = env.reset().get_state(agent.vision)
	while not env.terminal:
		env.render()
		game_state = env.action(agent(game_state))[3]
	env.render()
