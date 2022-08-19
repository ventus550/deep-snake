#!/usr/bin/env bash
SNAKE=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function install {
	rm -rf bin include lib lib64
	python -m venv $SNAKE
	echo "export PYTHONPYCACHEPREFIX=/tmp/DeepSnake" >> $SNAKE/bin/activate
	echo "source "$SNAKE"/snake.sh" >> $SNAKE/bin/activate
	source $SNAKE/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt
	ln -s $SNAKE/snake lib/python3.10/site-packages
	ln -s $SNAKE/QLearning lib/python3.10/site-packages
	ln -s $SNAKE/cli lib/python3.10/site-packages
	ln -s $SNAKE/bin/activate $SNAKE
}

function play {
	python $SNAKE/cli/play.py $1
}

function test {
	python $SNAKE/cli/test.py $1
}

function list {
	if [ $1 ]; then
		python -m json.tool $SNAKE/QLab/$1/agent.json
	else
		python $SNAKE/cli/list.py
	fi
}

function retrain {
	python $SNAKE/cli/train.py $1
}

function train {
	agent=$(find $SNAKE/QLab -type d -name $1) 
	if [ -f $agent/net ]; then
		echo "Model is already trained. Continue?"
		if [ ! $(read) ]; then
			retrain $1;
		fi
	else
		retrain $1
	fi
	test $1
}

function clean {
	find $SNAKE -type d -name __pycache__ -exec rm -r {} \+
	rm -rf share
}

"$@"