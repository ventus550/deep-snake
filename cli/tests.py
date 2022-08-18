from itertools import product
from QLearning import Agent
from snake import Environment
from utils import progress_bar, ftensor


def quickie(agent : Agent):
	env = agent.env
	game_state = env.reset().get_state(agent.vision)
	while not env.terminal:
		game_state = env.action(agent(game_state))[3]
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
	env = Environment(shape = (2*vis + 1,) * 2)
	fails = 0
	for n, t in enumerate(progress_bar(list(blocks(vis)), color="blue", desc="Testing collisions")):
		env.reset()
		env.map = t
		env.action(agent(t))
		fails += env.terminal
	return (n - fails) / n
