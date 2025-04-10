.PHONY: venv clean run analyze train all watch ml-watch

# Define Python command to use venv
PYTHON=./venv/bin/python

clean:
	rm -rf game_data/* && rm -rf analysis/*

run:
	$(PYTHON) src/main.py

analyze:
	$(PYTHON) src/analyze_game_data.py

train:
	$(PYTHON) src/train_agent_model.py

watch:
	$(PYTHON) src/main.py -i

watch-ml:
	$(PYTHON) src/main.py -i --strategy ml

all:
	$(MAKE) clean
	$(MAKE) run
	$(MAKE) analyze

# Create venv if it doesn't exist
venv:
	python3 -m venv venv
	./venv/bin/pip install -e .