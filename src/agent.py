# src/agent.py

from typing import TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from game import GameState

import random

# Possible actions for agents
WAIT = 0
CHANGE_SWITCH_STATE = 1
MOVE_UP = 2
MOVE_DOWN = 3
MOVE_RIGHT = 4
MOVE_LEFT = 5


class Agent:
    """Represents an independent agent that can move around the map"""

    def __init__(self, x: int, y: int, agent_id: int):
        """
        Initialize an agent at a specific position

        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            agent_id: Unique identifier for this agent
        """
        self.position = (x, y)
        self.id = agent_id
        self.alive = True

    def update(self, game_state: "GameState") -> None:
        """
        Get the agent's desired next action

        Args:
            game_state: Current state of the game for decision making
        """

        if game_state.is_trolley_collision(self.position):
            self.alive = False

        if not self.alive:
            return

        # For now, choose a random action
        action = random.choice([WAIT, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT])

        x, y = self.position
        new_x, new_y = x, y

        if action == WAIT:
            pass  # Do nothing
        elif action == CHANGE_SWITCH_STATE and game_state.is_on_switch(self.position):
            game_state.toggle_switch()
        elif action == MOVE_UP:
            new_y = y - 1
        elif action == MOVE_DOWN:
            new_y = y + 1
        elif action == MOVE_LEFT:
            new_x = x - 1
        elif action == MOVE_RIGHT:
            new_x = x + 1

        # only apply valid motion
        new_position = (new_x, new_y)
        if game_state.is_in_bounds(new_position):
            self.position = new_position
