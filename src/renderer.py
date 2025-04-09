# src/renderer.py

import curses
from typing import Dict, List, Tuple, Optional

from game import GameState

EMPTY = " "
TRACK = "="
SWITCH_UP = "/"
SWITCH_DOWN = "\\"
TROLLEY = "0"


class GameRenderer:
    """Handles all rendering logic for the game"""

    def __init__(self, stdscr):
        """Initialize the renderer with a curses screen and set up colors"""
        self.stdscr = stdscr
        self._setup_curses()

        # Define color mappings for different game elements
        self.color_map = {
            EMPTY: 0,  # Default
            TRACK: 1,  # White
            SWITCH_UP: 1,  # White (same as track)
            SWITCH_DOWN: 1,  # White (same as track)
            TROLLEY: 2,  # Red
        }

    def _setup_curses(self):
        """Set up curses display settings"""
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Track
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)  # Trolley

    def render_frame(self, game_state: "GameState") -> None:
        """Render a single frame of the game with the provided game state"""
        # Clear screen
        self.stdscr.clear()

        # Draw game grid
        self._draw_grid(game_state)

        # Draw status information
        self._draw_status(game_state)

        # Refresh the screen
        self.stdscr.refresh()

    def _draw_track(
        self, display_grid: List[List[str]], game_state: "GameState"
    ) -> None:
        """Draw the track elements on the display grid"""
        height = game_state.height
        width = game_state.width
        switch_x, switch_y = game_state.trolley_switch_position

        # Draw track on the left side up to the switch
        for x in range(switch_x):
            display_grid[switch_y][x] = TRACK

        # Upper branch (switch UP)
        for x in range(switch_x + 1, width):
            display_grid[switch_y - 1][x] = TRACK

        # Lower branch (switch DOWN)
        for x in range(switch_x + 1, width):
            display_grid[switch_y + 1][x] = TRACK

        # Add the switch at the branching point
        switch_char = SWITCH_UP if game_state.trolley_switch_state else SWITCH_DOWN
        display_grid[switch_y][switch_x] = switch_char

    def _draw_trolley(
        self, display_grid: List[List[str]], game_state: "GameState"
    ) -> None:
        """Draw the trolley on the display grid if it exists"""
        if game_state.trolley_position:
            x, y = game_state.trolley_position
            # Only draw the trolley if it's within the grid boundaries
            if 0 <= y < game_state.height and 0 <= x < game_state.width:
                display_grid[y][x] = TROLLEY

    def _draw_grid(self, game_state: "GameState") -> None:
        """Draw the game grid with all elements"""
        # Create a visual representation of the game state

        # Start with a clean grid filled with empty spaces
        height = game_state.height
        width = game_state.width
        display_grid = [[EMPTY for _ in range(width)] for _ in range(height)]

        self._draw_track(display_grid, game_state)
        self._draw_trolley(display_grid, game_state)

        # Render the display grid
        for y, row in enumerate(display_grid):
            for x, cell in enumerate(row):
                # Ensure cell is a string
                cell_str = str(cell)
                color_pair = self.color_map.get(cell_str, 0)
                try:
                    self.stdscr.addstr(
                        y,
                        x * 2,  # Double x for better aspect ratio
                        cell_str,
                        curses.color_pair(color_pair) if color_pair else 0,
                    )
                except curses.error:
                    # Handle potential out-of-bounds errors which can happen at screen edges
                    pass

    def _draw_status(self, game_state: "GameState") -> None:
        """Draw status information below the game grid"""
        try:
            # Basic game info
            self.stdscr.addstr(game_state.height + 1, 0, f"Turn: {game_state.turn}")

            # Display switch state
            switch_state = "UP" if game_state.trolley_switch_state else "DOWN"
            self.stdscr.addstr(game_state.height + 2, 0, f"Switch: {switch_state}")

            # Display trolley info
            trolley_info = (
                "Not on screen"
                if game_state.trolley_position is None
                else f"at {game_state.trolley_position}"
            )
            self.stdscr.addstr(game_state.height + 3, 0, f"Trolley: {trolley_info}")

            # Control instructions
            self.stdscr.addstr(
                game_state.height + 4, 0, "Controls: 'q' to quit, 's' to toggle switch"
            )
        except curses.error:
            # Handle potential out-of-bounds errors
            pass

    def get_key(self) -> int:
        """Get a key press from the user (non-blocking)"""
        self.stdscr.timeout(0)
        return self.stdscr.getch()

    def cleanup(self):
        """Clean up the curses environment"""
        curses.endwin()


def initialize_renderer():
    """Initialize curses and return a GameRenderer instance"""
    stdscr = curses.initscr()
    renderer = GameRenderer(stdscr)
    return renderer
