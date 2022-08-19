import json
from sys import argv
from itertools import product
from torch import randint, float32
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

def valid(t, vis, sides):
	return t[vis, vis] == 1 and 0 < sum(t[s] for s in sides) < 4

def rng(vis, sides):
	vsd = 2*vis+1
	count = 0
	print("Generating test cases..")
	while count < 100000:
		t = randint(0, 2, (vsd, vsd), dtype=float32)
		if valid(t, vis, sides):
			count += 1
			yield t

def combs(vis, sides):
	vsd = 2*vis+1
	cmbs = product( *(((1,0),) * vsd**2) )
	for c in cmbs:
		t = ftensor(c).reshape((vsd, vsd))
		if valid(t, vis, sides):
			yield t

def test_collisions(agent : Agent):
	vis = agent.vision
	vsd = 2*vis + 1
	sides = (
		(vis + 1, vis),
		(vis, vis + 1),
		(vis - 1, vis),
		(vis, vis - 1)
	)
	env = Environment(width = vsd, height = vsd, vision=vis)
	fails = 0
	blocks = combs(vis, sides) if vis == 1 else rng(vis, sides)
	for n, t in enumerate(progress_bar(list(blocks), color="blue", desc="Testing collisions")):
		env.reset()
		env.map = t
		env.walls = t
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
		"score variance": var,
		"walls": int(env.walls.count_nonzero()) > 0
	}
	outfile.write(json.dumps(data, indent=4))