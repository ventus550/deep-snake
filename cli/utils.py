import os
import json
from snake import *
from QLearning.utils import *
from QLearning import Agent, models

QLab = f"{os.getcwd()}/QLab"

def directory(dir_path):
	for entry in os.listdir(dir_path):
		yield f"{dir_path}/{entry}"

def isdir(entry):
	return os.path.isdir(entry)

def fetch(agent : str, data : str):
	return json.load(open(f"{QLab}/{agent}/{data}.json"))

def initialize(agent : str):
	data = fetch(agent, "agent")
	env = Environment(**data["Environment"])

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
