# 🚋 Trolley World

A multi-agent simulation of the classic trolley problem where autonomous agents with different personalities make moral decisions in a dynamic environment.

<p align="middle">
<video width="640" height="360" controls>
  <source src="./assets/game.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>
</p>

## About the Project

Trolley World is a simulation that brings the famous trolley problem to life with autonomous agents who have varying personality traits (altruism, risk aversion, and aggression). The simulation explores how these traits affect decision-making in ethical dilemmas.

### The Trolley Problem

The classic trolley problem is a thought experiment in ethics that poses a moral dilemma:

> A trolley is running out of control down a track. In its path are 5 people who have been tied to the track. You are standing next to a lever that controls a switch. If you pull the lever, the trolley will be redirected onto a side track, where one person is tied. What should you do?

This simulation expands on this problem by creating a dynamic environment where multiple agents interact with each other and must make decisions based on their individual personalities.

## Features

- **Multi-agent simulation** with autonomous agents making ethical decisions
- **Personality traits** influence agent behavior:
  - **Altruism**: Tendency to help others
  - **Risk aversion**: Caution around danger
  - **Aggression**: Willingness to harm others
- **Dynamic environment** with:
  - Trolley that follows tracks and can be redirected with a switch
  - Food resources that must be gathered to maintain energy
  - Social interactions between agents (gift energy, tie to tracks, untie prisoners)
- **Data collection** for analyzing agent behavior and decision patterns
- **Machine learning integration** for training and implementing AI decision-making models

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (install with `pip install -r requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - joblib
  - curses (included with Python on Unix/Linux/macOS; for Windows you may need `windows-curses`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/trolley-world.git
   cd trolley-world
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

To run the simulation in interactive mode with visualization:

```bash
python src/main.py --interactive
```

To run the simulation in batch mode for data collection:

```bash
python src/main.py
```

### Analyzing the Results

After running the simulation, you can analyze the collected data:

```bash
python src/analyze_game_data.py --data-dir game_data --output-dir analysis
```

This will generate various visualizations and statistics about agent behavior, decision patterns, and trolley dilemmas.

### Training Machine Learning Models

To train a machine learning model based on the collected data:

```bash
python src/train_agent_model.py --data-path analysis/training_data.csv --output-dir models
```

To use the trained model in the simulation, modify the agent initialization in `game.py` to use the `MLStrategy`.

## Project Structure

```
trolley-world/
├── src/
│   ├── agent.py             # Agent class and action definitions
│   ├── game.py              # Main game state and logic
│   ├── main.py              # Entry point for the simulation
│   ├── renderer.py          # Terminal-based visualization
│   ├── data_collector.py    # Data collection for analysis
│   ├── analyze_game_data.py # Data analysis and visualization
│   ├── train_agent_model.py # ML model training
│   ├── strategy.py          # Strategy interface
│   └── strategies/          # Agent decision-making strategies
│       ├── heuristic_strategy.py # Rule-based decision making
│       ├── ml_strategy.py        # ML-based decision making
│       └── random_strategy.py    # Random decision making
├── game_data/               # Stored game data (JSON)
├── analysis/                # Analysis output
├── models/                  # Trained ML models
└── requirements.txt
```

## Understanding the Data

### Game Records

Game records are stored as JSON files in the `game_data` directory with the following structure:

- `game_id`: Unique identifier for the game
- `game_config`: Configuration details (width, height, switch position)
- `agent_profiles`: Initial agent configurations and personality traits
- `turns`: Turn-by-turn state of the game including:
  - Agent positions, energy, scores
  - Trolley position and switch state
  - Affinity matrix (relationships between agents)
  - Actions taken by agents
- `final_scores`: Final scores of all agents
- `survival_turns`: Number of turns each agent survived

### Analysis Files

The analysis script generates several CSV files in the `analysis` directory:

- `agent_performance.csv`: Overall agent performance metrics
- `action_outcomes.csv`: Outcomes from different agent actions
- `social_interactions.csv`: Data about agent interactions
- `trolley_dilemmas.csv`: Information about trolley dilemma situations
- `training_data.csv`: Prepared data for ML model training

## Creating Custom Strategies

You can create custom agent strategies by implementing the `AgentStrategy` interface:

```python
from strategy import AgentStrategy

class MyCustomStrategy(AgentStrategy):
    def decide_action(self, agent, game_state):
        # Your logic here to decide the next action
        return Action.MOVE_UP  # Example action
```

## Trolley World Mechanics

### Agent Actions

Agents can perform the following actions:

- **Movement**: Up, down, left, right
- **Wait**: Do nothing for a turn
- **Toggle Switch**: Change the trolley track direction
- **Gift Energy**: Share energy with another agent
- **Tie to Tracks**: Tie another agent to the trolley tracks
- **Untie Prisoner**: Free an agent tied to the tracks

### Energy and Survival

- Agents start with 500 energy units
- Energy decreases by 2 units per turn
- Energy decreases by 1 unit per turn when tied to tracks
- Agents can gain energy by finding and consuming food
- When energy reaches 0, the agent dies

### Social Dynamics

- Agents have affinity scores toward other agents
- Affinities change based on interactions:
  - Gifting energy increases affinity
  - Tying to tracks decreases affinity
  - Untying from tracks increases affinity
- Affinity influences decision-making with the HeuristicStrategy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
