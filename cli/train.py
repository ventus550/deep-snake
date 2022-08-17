from sys import argv
from QLearning import schedulers
import json
from utils import initialize, fetch, ignored, QLab
from tests import test_collisions, test_scores

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

agent = initialize(target)
agent.load(f"{QLab}/{target}/net")
agent.train(live=True, episodes=50, scheduler=schedulers.zero)
# import play

# colis = test_collisions(agent)
# mn, mx, mean, var = test_scores(agent)
# with open(f"{QLab}/{target}/stats.json", "w") as outfile:
# 	data = {
# 		"collisions": colis,
# 		"min score": mn,
# 		"max score": mx,
# 		"mean score": mean,
# 		"score variance": var
# 	}
# 	outfile.write(json.dumps(data, indent=4))