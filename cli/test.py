import json
from sys import argv
from itertools import product
from QLearning import Agent
from snake import Environment
from utils import progress_bar, ftensor, initialize, ignored, QLab


def quickie(agent : Agent):
	env = agent.env
	all(agent.playoff())
	return env.score

def test_scores(agent : Agent):
	prg = progress_bar(range(100), color="blue", desc="Testing scores")
	games = [quickie(agent) for _ in prg]
	mean = sum(games)//len(games)
	return (
		min(games),
		max(games),
		mean,
		sum((g - mean)**2 for g in games) // (len(games) - 1)
	)		

def blocks(vis):
	vside = 2*vis+1
	sides = (
		(vis + 1, vis),
		(vis, vis + 1),
		(vis - 1, vis),
		(vis, vis - 1)
	)
	combs = product( *(((1,0),) * 9) )
	for c in combs:
		t = ftensor(c).reshape((vside, vside))
		if t[vis, vis] == 1 and 0 < sum(t[s] for s in sides) < 4:
			yield t

def test_collisions(agent : Agent):
	vis = agent.vision
	vsd = 2*vis + 1
	env = Environment(width = vsd, height = vsd)
	fails = 0
	for n, t in enumerate(progress_bar(list(blocks(vis)), color="blue", desc="Testing collisions")):
		env.reset()
		env.map = t
		env.action(agent(t))
		fails += env.terminal
	return (n - fails) / n

try:
	target = argv[-1]
	agent = initialize(target)
	with ignored(FileNotFoundError):
		agent.load(f"{QLab}/{target}/net")
	env = agent.env
except FileNotFoundError:
	exit("Agent does not exist")


colis = test_collisions(agent)
mn, mx, mean, var = test_scores(agent)
with open(f"{QLab}/{target}/stats.json", "w") as outfile:
	data = {
		"collisions": colis,
		"min score": mn,
		"max score": mx,
		"mean score": mean,
		"score variance": var
	}
	outfile.write(json.dumps(data, indent=4))