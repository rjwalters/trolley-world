# src/strategies/random.py

import random
from typing import TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from game import GameState
    from agent import Agent, Action

# Import Strategy base class
from strategy import AgentStrategy


class RandomStrategy(AgentStrategy):
    """
    Strategy that makes completely random decisions for agent actions.
    This mimics the original agent behavior.
    """

    def __init__(self):
        """Initialize the random strategy"""
        pass

    def decide_action(self, agent: "Agent", game_state: "GameState") -> "Action":
        """
        Choose a random action for the agent to take.

        Args:
            agent: The agent that needs to decide on an action
            game_state: Current state of the game

        Returns:
            A randomly selected Action
        """
        from agent import Action  # Import here to avoid circular imports

        # Choose a random action from all possible actions
        return random.choice(
            [
                Action.WAIT,
                Action.MOVE_UP,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
                Action.CHANGE_SWITCH_STATE,
                Action.GIFT_ENERGY,
                Action.TIE_TO_TRACKS,
                Action.UNTIE_PRISONER,
            ]
        )
