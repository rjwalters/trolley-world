# src/data_collector.py

import json
import os
import time
from typing import Dict, List, Any, Optional
import numpy as np

from agent import Agent, Action
from game import GameState


class GameRecord:
    """
    Class to store a single game's data for training purposes
    """

    def __init__(self, game_id: str):
        """
        Initialize a new game record

        Args:
            game_id: Unique identifier for this game
        """
        self.game_id = game_id
        self.game_config: Dict[str, Any] = {}
        self.turns: List[Dict[str, Any]] = []
        self.agent_profiles: Dict[int, Dict[str, Any]] = {}
        self.final_scores: Dict[int, int] = {}
        self.survival_turns: Dict[int, int] = {}

    def set_game_config(self, game_state: GameState) -> None:
        """
        Record initial game configuration

        Args:
            game_state: Current game state to extract configuration from
        """
        self.game_config = {
            "width": game_state.width,
            "height": game_state.height,
            "switch_position": game_state.trolley_switch_position,
            "num_agents": len(game_state.agents),
        }

    def record_agent_profiles(self, agents: List[Agent]) -> None:
        """
        Record the initial agent profiles including their strategies

        Args:
            agents: List of game agents
        """
        for agent in agents:
            if hasattr(agent.strategy, "altruism"):
                self.agent_profiles[agent.id] = {
                    "initial_position": agent.position,
                    "initial_energy": agent.energy,
                    "altruism": agent.strategy.altruism,
                    "risk_aversion": agent.strategy.risk_aversion,
                    "aggression": agent.strategy.aggression,
                }

    def record_turn(self, game_state: GameState) -> None:
        """
        Record data for a single game turn

        Args:
            game_state: Current game state
        """
        turn_data = {
            "turn_number": game_state.turn,
            "trolley_position": game_state.trolley_position,
            "switch_state": game_state.trolley_switch_state,
            "agents": [],
        }

        # Flatten the affinity matrix row by row
        affinity_matrix_flat = {}
        for i in range(len(game_state.agents)):
            for j in range(len(game_state.agents)):
                if i != j:  # Skip self-affinities
                    affinity_matrix_flat[f"{i}_{j}"] = int(
                        game_state.affinity_matrix[i, j]
                    )

        turn_data["affinity_matrix"] = affinity_matrix_flat

        # Record agent states
        for agent in game_state.agents:
            # Skip if agent was never alive (should never happen)
            if not hasattr(agent, "position"):
                continue

            agent_data = {
                "id": agent.id,
                "position": agent.position,
                "alive": agent.alive,
                "energy": agent.energy,
                "score": agent.score,
                "is_tied_to_tracks": agent.is_tied_to_tracks,
            }

            # Add the most recent action if available
            if agent.action_history:
                agent_data["last_action"] = agent.action_history[-1].name

            turn_data["agents"].append(agent_data)

        self.turns.append(turn_data)

    def record_game_end(self, game_state: GameState) -> None:
        """
        Record final game statistics

        Args:
            game_state: Final game state
        """
        for agent in game_state.agents:
            self.final_scores[agent.id] = agent.score
            # Record how long each agent survived
            last_alive_turn = game_state.turn
            for turn_idx in range(len(self.turns) - 1, -1, -1):
                turn = self.turns[turn_idx]
                agent_data = next(
                    (a for a in turn["agents"] if a["id"] == agent.id), None
                )
                if agent_data and not agent_data["alive"]:
                    last_alive_turn = turn["turn_number"]
                    break
            self.survival_turns[agent.id] = last_alive_turn

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the game record to a dictionary for serialization

        Returns:
            Dictionary representation of this game record
        """
        return {
            "game_id": self.game_id,
            "game_config": self.game_config,
            "agent_profiles": self.agent_profiles,
            "turns": self.turns,
            "final_scores": self.final_scores,
            "survival_turns": self.survival_turns,
        }


class DataCollector:
    """
    Collects and saves game data for future training
    """

    def __init__(self, output_dir: str = "game_data"):
        """
        Initialize the data collector

        Args:
            output_dir: Directory to save game records
        """
        self.output_dir = output_dir
        self.current_game: Optional[GameRecord] = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def start_new_game(self, game_state: GameState) -> None:
        """
        Start recording a new game

        Args:
            game_state: Initial game state
        """
        # Generate a unique game ID using timestamp
        game_id = f"game_{time.time_ns()}"
        self.current_game = GameRecord(game_id)

        # Record initial game configuration
        self.current_game.set_game_config(game_state)
        self.current_game.record_agent_profiles(game_state.agents)

    def record_turn(self, game_state: GameState) -> None:
        """
        Record data for the current turn

        Args:
            game_state: Current game state
        """
        if self.current_game is None:
            self.start_new_game(game_state)

        self.current_game.record_turn(game_state)

    def save_game_record(self, game_state: GameState) -> str:
        """
        Save the current game record to disk

        Args:
            game_state: Final game state

        Returns:
            Path to the saved record file
        """
        if self.current_game is None:
            raise RuntimeError("No game record to save")

        # Record final game statistics
        self.current_game.record_game_end(game_state)

        # Convert to dictionary and save as JSON
        record_dict = self.current_game.to_dict()

        # Create filename with game ID
        filename = f"{self.current_game.game_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Handle numpy types for JSON serialization
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(record_dict, f, cls=NumpyEncoder, indent=2)

        # Reset current game
        self.current_game = None

        return filepath

    def get_latest_records(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Load the n most recent game records

        Args:
            n: Number of records to load

        Returns:
            List of loaded game records
        """
        # List all JSON files in the output directory
        json_files = [f for f in os.listdir(self.output_dir) if f.endswith(".json")]

        # Sort by modification time (newest first)
        json_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)),
            reverse=True,
        )

        # Load the n most recent files
        records = []
        for i, filename in enumerate(json_files[:n]):
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "r") as f:
                records.append(json.load(f))

        return records
