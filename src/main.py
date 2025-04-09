# src/main.py

import curses
import time
from typing import Any

from game import GameState
from renderer import initialize_renderer


def main(stdscr: Any) -> None:
    """
    Main game loop function

    Args:
        stdscr: Curses standard screen object
    """
    # Initialize renderer and game state
    renderer = initialize_renderer()
    game_state = GameState()

    # Game loop
    running = True
    while running:
        # Handle input
        key = renderer.get_key()
        if key == ord("q"):
            running = False
            break
        elif key == ord("s"):
            game_state.toggle_switch()
        elif key == ord("r"):
            game_state.reset()

        # Update game state (move trolley, etc.)
        running = game_state.update()

        # Render frame
        renderer.render_frame(game_state)

        # Control update rate
        time.sleep(0.01)

    # Clean up curses when done
    renderer.cleanup()


if __name__ == "__main__":
    # Start game with curses wrapper
    curses.wrapper(main)
