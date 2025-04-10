# src/strategy.py

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from game import GameState
    from agent import Agent, Action


class AgentStrategy(ABC):
    """
    Abstract base class for agent decision-making strategies.
    Each strategy implementation provides a way to decide the next action
    for an agent given the current game state.
    """

    @abstractmethod
    def decide_action(self, agent: "Agent", game_state: "GameState") -> "Action":
        """
        Determine the next action for an agent to take.

        Args:
            agent: The agent that needs to decide on an action
            game_state: Current state of the game for decision making

        Returns:
            The Action the agent should take
        """
        pass

    def __str__(self) -> str:
        """Return a string representation of the strategy"""
        return self.__class__.__name__
