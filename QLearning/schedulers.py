from itertools import count
from math import exp

def scheduler(func):
	def init(decay):
		for step in count():
			yield max(0, func(decay, step))
	def inner(decay):
		return iter(init(1/decay))
	return inner

@scheduler
def linear(decay, step):
	return -decay * step + 1

@scheduler
def exponential(decay, step):
	return exp(-6 * decay * step)

@scheduler
def sigmoidal(decay, step):
	return 1 / (exp(10 * decay * step - 5) + 1)

@scheduler
def zero(decay, step):
	return 0

@scheduler
def one(decay, step):
	return 1
