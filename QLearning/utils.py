import torch
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextlib
from seaborn import set_theme
set_theme(style = "darkgrid", palette="dark")

@contextlib.contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

def get_device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
