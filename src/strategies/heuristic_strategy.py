# src/strategies/heuristic_strategy.py

import random
from typing import TYPE_CHECKING, List, Tuple

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from game import GameState
    from agent import Agent, Action

# Import Strategy base class
from strategy import AgentStrategy


class HeuristicStrategy(AgentStrategy):
    """
    Strategy that makes decisions based on heuristics and the current game state.
    Prioritizes survival, energy maintenance, and social interactions.
    """

    def __init__(self, altruism=0.5, risk_aversion=0.7, aggression=0.3):
        """
        Initialize the heuristic strategy with personality parameters.

        Args:
            altruism: How likely the agent is to help others (0.0-1.0)
            risk_aversion: How cautious the agent is about danger (0.0-1.0)
            aggression: How likely the agent is to harm others (0.0-1.0)
        """
        self.altruism = altruism
        self.risk_aversion = risk_aversion
        self.aggression = aggression

    def decide_action(self, agent: "Agent", game_state: "GameState") -> "Action":
        """
        Determine the next action for an agent based on heuristics.

        Args:
            agent: The agent that needs to decide on an action
            game_state: Current state of the game

        Returns:
            The Action the agent should take
        """
        from agent import Action  # Import here to avoid circular imports

        x, y = agent.position

        # Priority 1: Don't die! Avoid trolley if it's coming
        if game_state.trolley_position is not None:
            trolley_x, trolley_y = game_state.trolley_position

            # If we're on the tracks and the trolley is coming (to our left)
            if game_state.is_on_track(agent.position) and trolley_x < x:
                # Chance to not avoid based on risk aversion
                if random.random() > self.risk_aversion:
                    # This agent is a risk-taker!
                    pass
                else:
                    # Move away from tracks - try up or down
                    if y > 0:
                        return Action.MOVE_UP
                    elif y < game_state.height - 1:
                        return Action.MOVE_DOWN

        # Priority 2: If we're low on energy, find food
        if agent.energy < 100:
            # If there's food, go toward it
            if game_state.food_positions:
                closest_food = min(
                    game_state.food_positions,
                    key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y),
                )
                food_x, food_y = closest_food

                # Move toward the food
                if x < food_x:
                    return Action.MOVE_RIGHT
                elif x > food_x:
                    return Action.MOVE_LEFT
                elif y < food_y:
                    return Action.MOVE_DOWN
                elif y > food_y:
                    return Action.MOVE_UP

        # Priority 3: Social interactions
        # Find other agents at our position
        other_agents = [
            agent_other
            for agent_other in game_state.agents
            if agent_other.position == agent.position
            and agent_other.id != agent.id
            and agent_other.alive
        ]

        if other_agents:
            # Check if there's someone tied to tracks we could help
            tied_agents = [a for a in other_agents if a.is_tied_to_tracks]
            if tied_agents and game_state.is_on_track(agent.position):
                # More altruistic agents are more likely to help
                if random.random() < self.altruism:
                    return Action.UNTIE_PRISONER

            # If we have plenty of energy, consider gifting to build relationships
            if agent.energy > 200 and random.random() < self.altruism:
                return Action.GIFT_ENERGY

            # If we're on tracks and have an enemy, consider tying them
            enemies = [
                a
                for a in other_agents
                if game_state.affinity_matrix[a.id, agent.id] < -100
            ]
            if enemies and agent.energy > 150 and random.random() < self.aggression:
                return Action.TIE_TO_TRACKS

        # Priority 4: Control the switch when strategically valuable
        if game_state.is_on_switch(agent.position):
            switch_x, switch_y = game_state.trolley_switch_position

            # Check if there are agents on either branch that we want to save/harm
            agents_on_upper = [
                a
                for a in game_state.agents
                if a.alive
                and a.position[1] == switch_y - 1
                and a.position[0] > switch_x
            ]
            agents_on_lower = [
                a
                for a in game_state.agents
                if a.alive
                and a.position[1] == switch_y + 1
                and a.position[0] > switch_x
            ]

            if agents_on_upper or agents_on_lower:
                # Find average affinity on each branch
                upper_affinity = (
                    sum(
                        game_state.affinity_matrix[agent.id, a.id]
                        for a in agents_on_upper
                    )
                    if agents_on_upper
                    else 0
                )
                lower_affinity = (
                    sum(
                        game_state.affinity_matrix[agent.id, a.id]
                        for a in agents_on_lower
                    )
                    if agents_on_lower
                    else 0
                )

                # Switch to save friends and harm enemies
                if (
                    upper_affinity > lower_affinity
                    and not game_state.trolley_switch_state
                ):
                    return Action.CHANGE_SWITCH_STATE  # Switch to upper branch
                elif (
                    lower_affinity > upper_affinity and game_state.trolley_switch_state
                ):
                    return Action.CHANGE_SWITCH_STATE  # Switch to lower branch

        # Priority 5: Explore the environment - move around intelligently
        return random.choice(
            [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]
        )

    def __str__(self) -> str:
        """Return a descriptive string of this strategy with personality traits"""
        return f"HeuristicStrategy(altruism={self.altruism:.1f}, risk_aversion={self.risk_aversion:.1f}, aggression={self.aggression:.1f})"
