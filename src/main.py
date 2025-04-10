import curses
import time
from typing import Any
import argparse
import numpy as np

from game import GameState
from agent import Agent
from renderer import initialize_renderer
from data_collector import DataCollector

GAMES_TO_SIMULATE = 100


def apply_strategy_to_game(game_state: GameState, strategy_type: str) -> None:
    """
    Apply the specified strategy to all agents in the game state

    Args:
        game_state: Current game state
        strategy_type: Type of strategy to use ('heuristic' or 'ml')
    """
    if strategy_type == "ml":
        # Import ml strategy when needed
        from strategies.ml_strategy import MLStrategy

        print("Applying trained ML strategy...")

        # Try loading the model to verify it works
        try:
            test_strategy = MLStrategy(model_path="models/agent_model.joblib")
            print(f"Model loaded successfully: {test_strategy.model is not None}")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return  # Early return if model can't be loaded

        # Create new agents with ML strategy at same positions
        ml_agents = []
        for i, agent in enumerate(game_state.agents):
            x, y = agent.position
            strategy = MLStrategy(model_path="models/agent_model.joblib")
            ml_agents.append(Agent(x, y, i, strategy, initial_energy=agent.energy))

        # Replace agents in game state
        game_state.agents = ml_agents

        # Reinitialize affinity matrix to maintain the same size
        num_agents = len(game_state.agents)
        game_state.affinity_matrix = np.zeros((num_agents, num_agents), dtype=int)


def main(stdscr: Any, args: argparse.Namespace) -> None:
    """
    Main game loop function

    Args:
        stdscr: Curses standard screen object
        args: Command line arguments
    """

    # Initialize renderer and game state
    renderer = initialize_renderer()
    game_state = GameState(16, 16)

    # Apply the specified strategy to agents
    apply_strategy_to_game(game_state, args.strategy)

    # Initialize data collector if data collection is enabled
    data_collector = DataCollector()
    data_collector.start_new_game(game_state)

    games_completed = 0
    rendering = True

    interactive_mode = args.interactive

    while games_completed < GAMES_TO_SIMULATE:
        # Update game state
        game_continues = game_state.update()

        # Record data if collection is enabled
        if data_collector:
            data_collector.record_turn(game_state)

        # draw initial world state
        if rendering or interactive_mode:
            renderer.render_frame(game_state)
            time.sleep(0.05)

        rendering = False

        # Check if game has ended
        if not game_continues:
            # Save game record if data collection is enabled
            if data_collector:
                filepath = data_collector.save_game_record(game_state)
                print(f"Game {games_completed + 1} data saved to: {filepath}")

            # Reset game state
            game_state.reset()

            # Apply strategy again after reset
            apply_strategy_to_game(game_state, args.strategy)

            # Start recording the new game
            if data_collector:
                data_collector.start_new_game(game_state)

            games_completed += 1
            rendering = True

    # Clean up curses when done
    renderer.cleanup()
    print(f"{games_completed} games played and recorded.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Trolley World simulation.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode with visualization",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["heuristic", "ml"],
        default="heuristic",
        help="Strategy type for agents (heuristic or ml)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Start game with curses wrapper
    curses.wrapper(lambda stdscr: main(stdscr, args))
