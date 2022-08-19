import torch
from collections import namedtuple

R = '\033[1;31m'
G = '\033[1;32m'
Y = '\033[1;33m'
B = '\033[1;34m'
n = '\033[0m'

def colored(str : str, color : str):
	return f"{color}{str}{n}"

def red(str : str):
	return colored(str, R)

def green(str : str):
	return colored(str, G)

def yellow(str : str):
	return colored(str, Y)

def blue(str : str):
	return colored(str, B)

def ftensor(values):
	return torch.tensor(values, dtype=torch.float32)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
