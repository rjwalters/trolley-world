.PHONY: clean run analyze train all

clean:
	rm -rf game_data/* && rm -rf analysis/*

run:
	python src/main.py

analyze:
	python src/analyze_game_data.py

train: 
	python src/train_agent_model.py

all:
	$(MAKE) clean
	$(MAKE) run
	$(MAKE) analyze