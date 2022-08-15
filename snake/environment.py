import torch
from os import system
from random import randint
from itertools import product
from snake.utils import *
from time import sleep

class Environment:
	"""
	Stores and manages the game environment and all its agents.
	The game world is loaded from the file as a plain text during the objects initialization.
	Game objects are then discretized and split into separate layers (walls, coins, ghosts, distances)
	and can be accessed individually or as a tensor (channels).

	Attributes:
		vision		-- size of the visible state returned by the get_state() method

		score		-- the sum of obtained rewards

		shape		-- game's world width and height measured in game blocks

		snake		-- snake's head position
	"""
	empty = 0
	apple = -1
	snake_body = 1
	directions = torch.tensor((
		(+1, 0),
		(0, +1),
		(-1, 0),
		(0, -1)
	))

	def __init__(self, shape = (30, 30), vision = 1):
		shape = shape[::-1]
		self.shape = torch.tensor(shape)
		self.center = self.shape.div(2).long()
		self.snake = self.center.clone()
		self.map = torch.zeros(*self.shape)
		self.actn = 0
		self.terminal = False
		self.score = 0
		self.vision = vision
		self.rewards = {
			Environment.apple: 10,
			Environment.empty: 1,
			Environment.snake_body: 0
		}
		self.move_snake(torch.tensor((0,0)))
		self.spawn_apple()

	def reset(self):
		self.__init__(tuple(self.shape)[::-1], self.vision)
		return self

	def spawn_apple(self):
		assert not self.terminal
		y = randint(0, self.shape[0] - 1)
		x = randint(0, self.shape[1] - 1)
		if self.map[y, x]:
			self.spawn_apple()
		else:
			self.map[y, x] = Environment.apple

	def recenter(self):
		shifts = tuple(self.center - self.snake)
		return self.map.roll(shifts = shifts, dims = (0, 1))
	
	def get_state(self):
		v = self.vision
		y, x = self.center
		recentered = self.recenter()
		visible_state = recentered[ y - v : y + v + 1, x - v : x + v + 1 ]
		return visible_state

	def move_snake(self, direction):
		self.snake += direction
		self.snake %= self.shape
		pos = tuple(self.snake)
		tile = self.map[pos].item()
		self.map[pos] = Environment.snake_body
		return tile

	def interact(self, tile):
		if tile == Environment.snake_body:
			self.terminal = True
		elif tile == Environment.apple:
			self.spawn_apple()
		
	def sars_terminal(self, old_state):
		return old_state, ftensor(self.actn), ftensor(0), None

	def sars_nonterminal(self, old_state, reward):
		return old_state, ftensor(self.actn), ftensor(reward), self.get_state()

	def action(self, action : int):
		assert isinstance(action, int)
		old_state = self.get_state()
		if self.terminal:
			return self.sars_terminal(old_state)
		
		self.actn = action or self.actn
		direction = Environment.directions[self.actn]
		tile = self.move_snake(direction)
		self.interact(tile)
		reward = self.rewards[tile]
		self.score += reward
		return self.sars_nonterminal(old_state, reward)

	def tiles(self):
		width, height = self.shape
		for x, y in product(range(width), range(height)):
			yield x, y, self.map[x, y]

	def render(self, wait = 0.5):
		sleep(wait)
		system("clear")
		print(self)
		print()

	def __repr__(self):
		y, x = Environment.directions[self.actn]
		res = [blue(f"  Score: {self.score}  Moved: {x.item(), y.item()}")]
		repr = {
			Environment.empty: '.',
			Environment.snake_body: yellow('x'),
			Environment.apple: red('o')
		}
		if self.terminal:
			res.append(red("  terminal"))
		for _, y, token in self.tiles():
			if y == 0: res.append('\n\t')
			res.append(repr[int(token)])
		return "".join(res)