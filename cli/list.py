from utils import *

def load(data, key, fill):
	try:
		return data[key]
	except KeyError:
		return fill

def tc(tests):
	testr = f"{tests}%"
	if tests > 66:
		return green(testr)
	elif tests > 33:
		return yellow(testr)
	return red(testr)

def ls(path):
	name = path.split("/")[-1]
	try:
		stats = fetch(name, "stats")
		score = load(stats, "mean score", "???")
		walls = blue("*") if load(stats, "walls", False) else ""
		colis = int(load(stats, "collisions", 0) * 100)
		if len(name) > 10: name = name[:8] + ".."
		print(f"{blue(name):<21}  {tc(colis)}  {score}{walls}")
	except FileNotFoundError:
		print(f"{blue(name):<21}  missing data file")

for entry in directory(QLab):
	if isdir(entry): ls(entry)