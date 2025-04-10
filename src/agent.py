# src/agent.py

from typing import TYPE_CHECKING, List, Dict, Tuple, Set, Optional

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from game import GameState
    from strategy import AgentStrategy

import random
from enum import Enum, auto
from itertools import combinations


class Action(Enum):
    FAILED = auto()
    WAIT = auto()
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    MOVE_RIGHT = auto()
    MOVE_LEFT = auto()
    CHANGE_SWITCH_STATE = auto()
    GIFT_ENERGY = auto()
    TIE_TO_TRACKS = auto()
    UNTIE_PRISONER = auto()


class Agent:
    """Represents an independent agent that can move around the map"""

    def __init__(
        self,
        x: int,
        y: int,
        agent_id: int,
        strategy: Optional["AgentStrategy"] = None,
        initial_energy: int = 500,
    ):
        """
        Initialize an agent at a specific position.
        Args:
            x: Starting x coordinate.
            y: Starting y coordinate.
            agent_id: Unique identifier for this agent (start at zero).
            strategy: Decision-making strategy for this agent.
            initial_energy: Starting energy for the agent.
        """
        self.position = (x, y)
        self.id = agent_id
        self.alive = True
        self.score = 0
        self.energy = initial_energy
        self.action_history = []
        self.is_tied_to_tracks = False
        self.strategy = strategy

    def update(self, game_state: "GameState") -> None:
        """
        Get the agent's desired next action.
        Args:
            game_state: Current state of the game for decision making.
        """

        if game_state.is_trolley_collision(self.position):
            self.alive = False

        if self.energy <= 0:
            self.alive = False

        if not self.alive:
            return

        # If this agent is tied to tracks, it can't take any action
        if self.is_tied_to_tracks:
            self.energy -= 1  # Lose energy more slowly when tied
            self.action_history.append(Action.WAIT)
            return

        # Check for food at current position and consume it
        energy_gained = game_state.consume_all_food(self.position)
        if energy_gained > 0:
            self.energy += energy_gained
            self.score += energy_gained  # Also increase score when food is found

        self.energy -= 2

        # Get the action from the agent's strategy
        if self.strategy:
            action = self.strategy.decide_action(self, game_state)
        else:
            # Default to WAIT if no strategy is set
            action = Action.WAIT

        x, y = self.position
        new_x, new_y = x, y

        # Handle the actions that require us to be on the same square as another agent
        if (
            action == Action.GIFT_ENERGY
            or action == Action.TIE_TO_TRACKS
            or action == Action.UNTIE_PRISONER
        ):
            # Find other agents at our position
            other_agents = [
                agent
                for agent in game_state.agents
                if agent.position == self.position
                and agent.id != self.id
                and agent.alive
            ]

            if action == Action.GIFT_ENERGY and other_agents and self.energy > 50:
                # Gift energy to the agent who likes us the most
                target_agent = max(
                    other_agents,
                    key=lambda a: game_state.affinity_matrix[a.id, self.id],
                )

                target_agent.energy += 100
                self.energy -= 10

                # Improve affinity between agents
                game_state.affinity_matrix[target_agent.id, self.id] += 500

                # Record action
                self.action_history.append(action)
                return

            elif action == Action.TIE_TO_TRACKS and other_agents and self.energy > 100:
                # Find the weakest agent on our square
                target_agent = min(other_agents, key=lambda a: a.energy)

                if self.energy > target_agent.energy:
                    # Get our current position
                    current_x, current_y = self.position
                    switch_x, switch_y = game_state.trolley_switch_position

                    # send to main track before switch
                    if current_x <= switch_x:
                        nearest_track_y = switch_y
                    else:
                        above_switch = (current_y - switch_y) < 0
                        if above_switch:
                            nearest_track_y = switch_y - 1
                        else:
                            nearest_track_y = switch_y + 1
                    # Tie agent to tracks!
                    target_agent.position = (current_x, nearest_track_y)
                    target_agent.is_tied_to_tracks = True
                    # Cost energy to tie someone
                    self.energy -= 50
                    # Decrease affinity between these agents
                    game_state.affinity_matrix[target_agent.id, self.id] = -1000
                    # Record action
                    self.action_history.append(action)
                    return

            elif action == Action.UNTIE_PRISONER and other_agents:
                # Find all tied agents
                tied_agents = [a for a in other_agents if a.is_tied_to_tracks]

                if tied_agents:
                    # Find the tied agent with highest affinity to us
                    target_agent = max(
                        tied_agents,
                        key=lambda a: game_state.affinity_matrix[a.id, self.id],
                    )

                    target_agent.is_tied_to_tracks = False
                    game_state.affinity_matrix[target_agent.id, self.id] += 1000
                    # Record action
                    self.action_history.append(action)
                    return

        if action == Action.WAIT:
            pass  # Do nothing
        elif action == Action.CHANGE_SWITCH_STATE and game_state.is_on_switch(
            self.position
        ):
            game_state.toggle_switch()
        elif action == Action.MOVE_UP:
            new_y = y - 1
        elif action == Action.MOVE_DOWN:
            new_y = y + 1
        elif action == Action.MOVE_LEFT:
            new_x = x - 1
        elif action == Action.MOVE_RIGHT:
            new_x = x + 1

        # only apply valid motion
        new_position = (new_x, new_y)
        if game_state.is_in_bounds(new_position):
            self.energy -= 2
            self.position = new_position
            self.action_history.append(action)
        else:
            self.action_history.append(Action.FAILED)


def find_intersections(agents: List[Agent]) -> Set[Tuple[int, int]]:
    """
    Finds intersections where more than one agent is at the same position and returns the IDs of these agents.
    Args:
        agents: List of Agent objects.
    Returns:
        A set of tuples, where each tuple contains the IDs of agents that intersect at a position.
    """
    position_to_ids: Dict[Tuple[int, int], List[int]] = {}

    # Map each position to a list of agent IDs at that position
    for agent in agents:
        if agent.position in position_to_ids:
            position_to_ids[agent.position].append(agent.id)
        else:
            position_to_ids[agent.position] = [agent.id]

    # Create a set to store all pairwise intersections
    intersections = set()

    # For each list of IDs in the dictionary, find all combinations of pairs
    for ids in position_to_ids.values():
        if len(ids) > 1:
            # Generate all unique pairs (combinations of 2) from ids
            for pair in combinations(ids, 2):
                intersections.add(pair)

    return intersections
