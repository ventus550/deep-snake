PWD = $(shell pwd)

run:
	find . -type f -name $(run) -exec python {} \+

install:
	( \
		python -m venv .; \
		source ./bin/activate; \
		pip install --upgrade pip; \
		pip install -r requirements.txt; \
		ln -s ${PWD}/snake lib/python3.10/site-packages; \
		ln -s ${PWD}/QLearning lib/python3.10/site-packages; \
	)

clean:
	find . -type d -name __pycache__ -exec rm -r {} \+
	rm -rf share

test:
