run:
	find . -type f -name $(run) -exec python {} \+

install:
	(                                       \
		python -m venv .;                   \
		source ./bin/activate;              \
		pip install --upgrade pip;          \
		pip install -r requirements.txt;    \
	)

clean:
	find . -type d -name __pycache__ -exec rm -r {} \+
	rm -rf share