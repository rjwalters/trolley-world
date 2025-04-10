# src/strategies/ml_strategy.py

import os
import joblib
import numpy as np
from typing import TYPE_CHECKING, Dict, Any, List

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from game import GameState
    from agent import Agent, Action

# Import Strategy base class
from strategy import AgentStrategy


class MLStrategy(AgentStrategy):
    """
    Strategy that makes decisions based on a trained machine learning model.
    """

    def __init__(self, model_path: str = "models/agent_model.joblib"):
        """
        Initialize the ML strategy with a trained model.

        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None

        # Load the model if it exists
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None

        # Fallback parameters if model fails
        self.altruism = 0.5
        self.risk_aversion = 0.7
        self.aggression = 0.3

    def decide_action(self, agent: "Agent", game_state: "GameState") -> "Action":
        """
        Determine the next action for an agent based on the trained model.

        Args:
            agent: The agent that needs to decide on an action
            game_state: Current state of the game

        Returns:
            The Action the agent should take
        """
        from agent import Action  # Import here to avoid circular imports

        # If model is not loaded, use HeuristicStrategy
        if self.model is None:
            from strategies.heuristic_strategy import HeuristicStrategy

            fallback = HeuristicStrategy(
                altruism=self.altruism,
                risk_aversion=self.risk_aversion,
                aggression=self.aggression,
            )
            return fallback.decide_action(agent, game_state)

        # Extract features from current game state
        features = self._extract_features(agent, game_state)

        # Convert features to DataFrame for prediction
        import pandas as pd

        features_df = pd.DataFrame([features])

        # Make prediction
        try:
            action_name = self.model.predict(features_df)[0]
            return getattr(Action, action_name)
        except Exception as e:
            print(f"Prediction error: {e}")
            return Action.WAIT

    def _extract_features(
        self, agent: "Agent", game_state: "GameState"
    ) -> Dict[str, Any]:
        """
        Extract features from game state for model prediction.

        Args:
            agent: The agent
            game_state: Current game state

        Returns:
            Dictionary of features
        """
        features = {}

        # Agent features
        features["agent_energy"] = agent.energy
        features["agent_tied"] = int(agent.is_tied_to_tracks)
        features["agent_x"], features["agent_y"] = agent.position

        # Agent personality (not used by model but included for consistency)
        features["agent_altruism"] = self.altruism
        features["agent_risk_aversion"] = self.risk_aversion
        features["agent_aggression"] = self.aggression

        # Trolley information
        if game_state.trolley_position:
            features["trolley_x"], features["trolley_y"] = game_state.trolley_position
            features["distance_to_trolley"] = abs(
                features["agent_x"] - features["trolley_x"]
            ) + abs(features["agent_y"] - features["trolley_y"])
            features["trolley_present"] = 1
        else:
            features["trolley_x"], features["trolley_y"] = -1, -1
            features["distance_to_trolley"] = -1
            features["trolley_present"] = 0

        # Switch information
        switch_x, switch_y = game_state.trolley_switch_position
        features["switch_x"], features["switch_y"] = switch_x, switch_y
        features["switch_state"] = int(game_state.trolley_switch_state)
        features["distance_to_switch"] = abs(features["agent_x"] - switch_x) + abs(
            features["agent_y"] - switch_y
        )
        features["is_at_switch"] = int(features["distance_to_switch"] == 0)

        # Track information - is agent on track?
        on_track = int(game_state.is_on_track(agent.position))
        features["on_track"] = on_track

        # Food information - simplified
        features["food_edge_y"] = (
            0 if game_state.turn % 200 < 100 else game_state.height - 1
        )
        features["distance_to_food_edge"] = abs(
            agent.position[1] - features["food_edge_y"]
        )

        # Nearby agents
        nearby_agents = []
        for other_agent in game_state.agents:
            if other_agent.id != agent.id and other_agent.alive:
                other_x, other_y = other_agent.position
                agent_x, agent_y = agent.position
                manhattan_dist = abs(agent_x - other_x) + abs(agent_y - other_y)
                if manhattan_dist <= 3:  # Consider agents within distance 3
                    nearby_agents.append(
                        {
                            "id": other_agent.id,
                            "distance": manhattan_dist,
                            "energy": other_agent.energy,
                            "tied": int(other_agent.is_tied_to_tracks),
                        }
                    )

        features["num_nearby_agents"] = len(nearby_agents)

        # Co-located agents
        colocated_agents = [a for a in nearby_agents if a["distance"] == 0]
        features["num_colocated_agents"] = len(colocated_agents)

        if colocated_agents:
            features["colocated_min_energy"] = min(
                a["energy"] for a in colocated_agents
            )
            features["colocated_max_energy"] = max(
                a["energy"] for a in colocated_agents
            )
            features["colocated_tied_agents"] = sum(a["tied"] for a in colocated_agents)
        else:
            features["colocated_min_energy"] = -1
            features["colocated_max_energy"] = -1
            features["colocated_tied_agents"] = 0

        # Affinity information
        affinity_sums = 0
        affinity_count = 0

        for i in range(len(game_state.agents)):
            if i != agent.id:
                affinity_sums += game_state.affinity_matrix[agent.id, i]
                affinity_count += 1

        features["avg_affinity_to_others"] = affinity_sums / max(affinity_count, 1)

        return features

    def __str__(self) -> str:
        """Return a string representation of the strategy"""
        return f"MLStrategy(model={self.model_path})"
