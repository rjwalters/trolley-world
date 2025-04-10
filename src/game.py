# src/game.py

from typing import Tuple, Optional, List
import random
import numpy as np

from agent import Agent, find_intersections
from strategy import AgentStrategy
from strategies.heuristic_strategy import HeuristicStrategy

# Number of turns to wait before spawning a new trolley
TROLLEY_SPAWN_INTERVAL = 100

# Number of agents at start
DEFAULT_NUMBER_OF_AGENTS = 10

# Number of turns to play with only one agent before ending the game
SINGLE_AGENT_GAME_END = 100


class GameState:
    """Manages the state of the trolley problem game"""

    def __init__(
        self,
        width: int = 40,
        height: int = 20,
        switch_x: Optional[int] = None,
        switch_y: Optional[int] = None,
        agents: List[Agent] = [],
    ):
        """Initialize the game state with default dimensions and customizable switch position"""
        self.width: int = width
        self.height: int = height
        self.turn: int = 0
        self.trolley_switch_state: bool = True  # Switch starts in UP position
        self.set_switch_position(switch_x, switch_y)

        # Start with a trolley on the tracks
        _, switch_y = self.trolley_switch_position
        self.trolley_position: Optional[Tuple[int, int]] = (0, switch_y)

        # add agents in random positions
        self.agents = []
        if len(agents) == 0:
            self.spawn_agents(DEFAULT_NUMBER_OF_AGENTS)
        else:
            self.agents = agents

        # Initialize affinity matrix
        num_agents = len(self.agents)
        self.affinity_matrix = np.zeros((num_agents, num_agents), dtype=int)

        self.food_positions: List[Tuple[int, int]] = []
        self.food_energy_value = 100
        self.spawn_food(bottom=False)

        self.turns_with_one_agent = 0

    def set_switch_position(self, x: Optional[int], y: Optional[int]) -> None:
        """Set the switch position to the specified coordinates"""

        # Use provided switch position or default to middle of screen
        if x is None:
            x = self.width // 2
        if y is None:
            y = self.height // 2

        # Ensure switch position is within bounds
        x = max(1, min(x, self.width - 2))  # Leaves room for branches
        y = max(1, min(y, self.height - 2))  # Leaves room for branches

        self.trolley_switch_position = (x, y)

    def toggle_switch(self) -> None:
        """Toggle the trolley switch between UP and DOWN positions"""
        self.trolley_switch_state = not self.trolley_switch_state

    def update_trolley(self) -> None:
        """Update the trolley position"""
        if self.trolley_position is None:
            return

        # Move trolley one position to the right
        x, y = self.trolley_position
        new_x: int = x + 1

        # Check if trolley moves past the switch point
        switch_x, switch_y = self.trolley_switch_position

        if x < switch_x and new_x >= switch_x:
            # Determine which path to take based on switch position
            if self.trolley_switch_state:  # Switch is UP
                new_y = switch_y - 1
            else:  # Switch is DOWN
                new_y = switch_y + 1
        else:
            new_y = y

        # Check if trolley is out of bounds (reached the end)
        if new_x >= self.width:
            self.trolley_position = None  # Remove trolley
        else:
            self.trolley_position = (new_x, new_y)

    def spawn_trolley(self) -> None:
        """spawn the trolley at the starting position"""
        _, switch_y = self.trolley_switch_position
        self.trolley_position = (0, switch_y)

    def update(self) -> bool:
        """
        Update the game state for one frame

        Returns:
            bool: False if game is over, True if game continues
        """

        self.turn += 1

        # Update existing trolley if present
        self.update_trolley()

        # Check if we need to spawn new food
        if not self.food_positions:
            # Alternate between top and bottom
            spawn_at_bottom = (
                self.turn % (TROLLEY_SPAWN_INTERVAL * 2) >= TROLLEY_SPAWN_INTERVAL
            )
            self.spawn_food(bottom=spawn_at_bottom)

        # affinity decays over time
        self.affinity_matrix -= 1

        # update the agents
        for agent in self.agents:
            agent.update(self)
            if agent.alive:
                # update this agents score based on how popular it is
                affinity_sum = np.sum(self.affinity_matrix[agent.id, :])
                agent.score += affinity_sum

        # If the trolley isn't on screen, check if it's time to spawn a new one
        if self.trolley_position is None and self.turn % TROLLEY_SPAWN_INTERVAL == 0:
            self.spawn_trolley()

        living_agents = [agent for agent in self.agents if agent.alive]

        if len(living_agents) == 0:
            return False

        if len(living_agents) == 1:
            self.turns_with_one_agent += 1

            if self.turns_with_one_agent > SINGLE_AGENT_GAME_END:
                return False

        # Always continue running
        return True

    def is_in_bounds(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position is within the game boundaries

        Args:
            position: A tuple (x, y) representing the position to check

        Returns:
            True if the position is within bounds, False otherwise
        """
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_trolley_collision(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position collides with the trolley

        Args:
            position: A tuple (x, y) representing the position to check

        Returns:
            True if the position collides with the trolley, False otherwise
        """
        if self.trolley_position is None:
            return False

        return position == self.trolley_position

    def is_on_switch(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position is on the switch

        Args:
            position: A tuple (x, y) representing the position to check

        Returns:
            True if the position is on the switch, False otherwise
        """
        x, y = position
        switch_x, switch_y = self.trolley_switch_position
        return x == switch_x and y == switch_y

    def is_on_track(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position is on any part of the track

        Args:
            position: A tuple (x, y) representing the position to check

        Returns:
            True if the position is on a track, False otherwise
        """
        x, y = position
        switch_x, switch_y = self.trolley_switch_position

        # Main track before switch
        if y == switch_y and x < switch_x:
            return True

        # Upper branch (after switch)
        if y == switch_y - 1 and x > switch_x:
            return True

        # Lower branch (after switch)
        if y == switch_y + 1 and x > switch_x:
            return True

        # Switch position
        if x == switch_x and y == switch_y:
            return True

        return False

    def consume_all_food(self, position: Tuple[int, int]) -> int:
        """
        Check if the given position is on top of food and consume it

        Args:
            position: A tuple (x, y) representing the position to check

        Returns:
            int: the energy change from eating all food at this position (0 if no food)
        """
        _x, y = position

        # Quick check: food only spawns on top (y=0) or bottom (y=height-1) rows
        if y != 0 and y != self.height - 1:
            return 0

        # Now that we know the position is on a food-spawning row, check if food exists there
        if position in self.food_positions:
            # Remove food from the list
            self.food_positions.remove(position)
            return self.food_energy_value

        return 0

    def spawn_agents(self, num_agents: int) -> None:
        """
        Spawn the specified number of agents at random positions

        Args:
            num_agents: Number of agents to spawn
        """
        self.agents = []

        for i in range(num_agents):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            # Select a strategy based on the distribution

            strategy = HeuristicStrategy(
                altruism=random.uniform(0.1, 0.9),
                risk_aversion=random.uniform(0.1, 0.9),
                aggression=random.uniform(0.1, 0.9),
            )

            # Create the agent with the selected strategy
            self.agents.append(Agent(x, y, i, strategy))

    def spawn_food(self, bottom: bool = False) -> None:
        """
        Spawn food at either the top or bottom edge.
        Args:
            bottom: If True, spawn at bottom edge; otherwise at top edge.
        """
        # Clear existing food
        self.food_positions = []

        # Determine y-coordinate based on spawn location
        y = self.height - 1 if bottom else 0

        # Count living agents
        num_living_agents = sum(1 for agent in self.agents if agent.alive)

        # Generate random x positions (at most num_living_agents - 1 food items)
        if num_living_agents > 1:
            food_count = num_living_agents - 1
            x_positions = random.sample(range(self.width), min(food_count, self.width))

            # Create and add new food items
            for x in x_positions:
                self.food_positions.append((x, y))

    def reset(self) -> None:
        """Reset the game state to initial conditions"""
        self.turn = 0

        # Set random switch position
        random_x = random.randint(5, self.width - 5)
        random_y = random.randint(3, self.height - 3)
        self.set_switch_position(random_x, random_y)

        # Reset agents
        self.spawn_agents(DEFAULT_NUMBER_OF_AGENTS)

        # Reinitialize affinity matrix
        num_agents = len(self.agents)
        self.affinity_matrix = np.zeros((num_agents, num_agents), dtype=int)

        # Reset and spawn trolley
        self.spawn_trolley()

        # Reset and spawn food
        self.food_positions = []
        self.spawn_food(bottom=False)
