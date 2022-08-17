import os
import contextlib
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from snake import Environment
from QLearning import Agent, models


R = '\033[1;31m'
G = '\033[1;32m'
Y = '\033[1;33m'
B = '\033[1;34m'
n = '\033[0m'
QLab = f"{os.getcwd()}/QLab"

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
	
@contextlib.contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass

def directory(dir_path):
	for entry in os.listdir(dir_path):
		yield f"{dir_path}/{entry}"

def isdir(entry):
	return os.path.isdir(entry)

def fetch(agent : str, data : str):
	return json.load(open(f"{QLab}/{agent}/{data}.json"))

def progress_bar(iterable, disabled = False, color="yellow", desc=None):
	return tqdm(iterable, disable=disabled, colour=color, desc=desc)

def quickplot(*values, legend = [], ylabel = "", path = "./quickplot"):
	legendary = len(legend) == len(values)
	for i, v in enumerate(values):
		plt.plot(v, label = legend[i] if legendary else "")
	if legendary:
		plt.legend()
	plt.tight_layout(pad=3)
	plt.ylabel(ylabel)
	plt.savefig(path)
	plt.clf()

def initialize(agent : str):
	data = fetch(agent, "agent")
	env = Environment(
		shape = (
			data["Environment"]["width"],
			data["Environment"]["height"]
		)
	)

	model = models.__dict__[data["model"]]
	net = model(data["vision"])
	
	agent = Agent(
		net, env,
		learning_rate=data["learning_rate"],
		gamma=data["gamma"],
		memory=data["memory"],
		vision=data["vision"]
	)
	return agent
