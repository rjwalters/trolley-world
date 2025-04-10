# src/main.py

import curses
import time
from typing import Any
import argparse

from game import GameState
from renderer import initialize_renderer
from data_collector import DataCollector

GAMES_TO_SIMULATE = 100


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
            time.sleep(0.01)

        rendering = False

        # Check if game has ended
        if not game_continues:
            # Save game record if data collection is enabled
            if data_collector:
                filepath = data_collector.save_game_record(game_state)
                print(f"Game {games_completed + 1} data saved to: {filepath}")

            # Reset game state
            game_state.reset()

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

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Start game with curses wrapper
    curses.wrapper(lambda stdscr: main(stdscr, args))
