import torch

R = '\033[1;31m'
Y = '\033[1;33m'
B = '\033[1;34m'
n = '\033[0m'

def colored(str : str, color : str):
	return f"{color}{str}{n}"

def red(str : str):
	return colored(str, R)

def yellow(str : str):
	return colored(str, Y)

def blue(str : str):
	return colored(str, B)

def ftensor(values):
	return torch.tensor(values, dtype=torch.float32)