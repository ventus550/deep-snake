import torch
from os import system
from random import randint
from random import random
from itertools import product
from time import sleep
from snake.utils import *

class Environment:
	"""
	Stores and manages the game environment.

	Attributes:
		score		-- the sum of obtained rewards

		shape		-- game's world width and height measured in game blocks

		snake		-- snake's head position
	"""
	empty = 0
	apple = -1
	tail = 1
	directions = torch.tensor((
		(+1, 0),
		(0, +1),
		(-1, 0),
		(0, -1)
	))

	def __init__(self, width = 30, height = 30, apples = 1, walls = False, vision = 1):
		self.shape = torch.tensor((height, width))
		self.center = self.shape.div(2).long()
		self.snake = self.center.clone()
		self.vision = vision
		self.apples = apples
		self.map = torch.zeros(*self.shape)
		self.walls = self.map.clone()
		self.terminal = False
		self.rewards = {
			Environment.apple: 10,
			Environment.empty: 1,
			Environment.tail: 0
		}
		self.actn = self.score = self.objects = 0
		self.move_snake(torch.tensor((0,0)))
		for y, x, _ in self.tiles():
			if walls and random() < 0.1:
				self.put(y, x, object = Environment.tail)
				self.walls[y, x] = 1

		for _ in range(apples):
			self.spawn_apple()

	def __len__(self):
		return (self.shape[0] * self.shape[1]).item()

	def reset(self):
		self.__init__(
			width = self.shape[1],
			height = self.shape[0],
			apples = self.apples,
			vision = self.vision,
			walls = bool(int(self.walls.count_nonzero()))
		)
		return self

	def put(self, y, x, object=0):
		tile = self.map[y, x].item()
		self.objects += tile == 0
		# self.objects -= tile != 0 and object == 0
		self.map[y, x] = object

	def spawn_apple(self):
		assert not self.terminal
		if self.objects >= len(self): return
		y = randint(0, self.shape[0] - 1)
		x = randint(0, self.shape[1] - 1)
		if self.map[y, x]:
			self.spawn_apple()
		else:
			self.put(y, x, object = Environment.apple)

	def recenter(self):
		shifts = tuple(self.center - self.snake)
		return self.map.roll(shifts = shifts, dims = (0, 1))
	
	def get_state(self):
		(y, x), v = self.center, self.vision
		recentered = self.recenter()
		visible_state = recentered[ y - v : y + v + 1, x - v : x + v + 1 ]
		return visible_state

	def move_snake(self, direction):
		self.snake += direction
		self.snake %= self.shape
		pos = tuple(self.snake)
		tile = self.map[pos].item()
		self.put(*pos, object = Environment.tail)
		return tile

	def interact(self, tile):
		if tile == Environment.tail:
			self.terminal = True
		elif tile == Environment.apple:
			self.spawn_apple()
		
	def sars_terminal(self, old_state):
		return Transition(old_state, ftensor(self.actn), ftensor(0), None)

	def sars_nonterminal(self, old_state, reward):
		return Transition(old_state, ftensor(self.actn), ftensor(reward), self.get_state())

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
		for y, x in product(range(width), range(height)):
			yield y, x, self.map[y, x]

	def render(self, wait = 0.5):
		sleep(wait)
		system("clear")
		print(self)
		print()

	def __repr__(self):
		repr = {
			Environment.empty: '.',
			Environment.tail: yellow('*'),
			Environment.apple: red('o')
		}
		y, x = Environment.directions[self.actn]
		res = [blue(f"  Score: {self.score}")]
		if self.terminal:
			res.append(red("  terminal"))
		for x, y, token in self.tiles():
			if y == 0:
				res.append('\n\t')
			if tuple(self.snake) == (x, y):
				res.append(green("*"))
			elif self.walls[x, y]:
				res.append(blue("x"))
			else:
				res.append(repr[int(token)])
		return "".join(res)