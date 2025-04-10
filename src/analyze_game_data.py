# src/analyze_game_data.py

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from collections import Counter


def load_game_records(data_dir: str, max_records: int = None) -> List[Dict[str, Any]]:
    """
    Load game records from the specified directory

    Args:
        data_dir: Directory containing game record JSON files
        max_records: Maximum number of records to load (None = all)

    Returns:
        List of loaded game records
    """
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    # Sort by modification time (newest first)
    json_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True
    )

    if max_records:
        json_files = json_files[:max_records]

    records = []
    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            records.append(json.load(f))

    return records


def extract_agent_performance(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract agent performance metrics from game records

    Args:
        records: List of game records

    Returns:
        DataFrame containing agent performance metrics
    """
    all_agents_data = []  # Renamed to avoid conflict with the inner variable

    for record in records:
        game_id = record["game_id"]
        for agent_id, profile in record["agent_profiles"].items():
            agent_id = int(agent_id)

            # Get agent's final score and survival turns
            final_score = record["final_scores"].get(str(agent_id), 0)
            survival_turns = record["survival_turns"].get(str(agent_id), 0)

            # Get agent strategy parameters
            altruism = profile.get("altruism", None)
            risk_aversion = profile.get("risk_aversion", None)
            aggression = profile.get("aggression", None)

            # Count action frequencies
            action_counts = Counter()
            for turn in record["turns"]:
                for agent_turn_data in turn["agents"]:  # Renamed to avoid conflict
                    if (
                        agent_turn_data["id"] == agent_id
                        and "last_action" in agent_turn_data
                    ):
                        action_counts[agent_turn_data["last_action"]] += 1

            # Calculate percentage of times agent was on tracks
            on_tracks_count = 0
            total_turns_alive = 0
            for turn in record["turns"]:
                for agent_turn_data in turn["agents"]:  # Renamed to avoid conflict
                    if agent_turn_data["id"] == agent_id and agent_turn_data["alive"]:
                        total_turns_alive += 1
                        # Check if position is on tracks (simplified check)
                        switch_position = record["game_config"]["switch_position"]
                        agent_position = agent_turn_data["position"]

                        switch_x, switch_y = switch_position
                        agent_x, agent_y = agent_position

                        if (
                            (agent_y == switch_y and agent_x <= switch_x)
                            or (
                                agent_y in (switch_y - 1, switch_y + 1)
                                and agent_x > switch_x
                            )
                            or (agent_x == switch_x and agent_y == switch_y)
                        ):
                            on_tracks_count += 1

            tracks_percentage = on_tracks_count / max(total_turns_alive, 1) * 100

            # Create row
            row = {
                "game_id": game_id,
                "agent_id": agent_id,
                "final_score": final_score,
                "survival_turns": survival_turns,
                "altruism": altruism,
                "risk_aversion": risk_aversion,
                "aggression": aggression,
                "tracks_percentage": tracks_percentage,
            }

            # Add action frequencies
            for action, count in action_counts.items():
                row[f"action_{action}"] = count

            all_agents_data.append(
                row
            )  # Append to the list intended for all agent data

    return pd.DataFrame(all_agents_data)


def analyze_action_outcomes(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze the outcomes of agent actions

    Args:
        records: List of game records

    Returns:
        DataFrame containing action outcome data
    """
    action_data = []

    for record in records:
        game_id = record["game_id"]

        # Process each turn except the last one
        for i in range(len(record["turns"]) - 1):
            current_turn = record["turns"][i]
            next_turn = record["turns"][i + 1]

            # Process each agent
            for agent_data in current_turn["agents"]:
                agent_id = agent_data["id"]

                # Skip if no action was taken
                if "last_action" not in agent_data:
                    continue

                action = agent_data["last_action"]
                position = tuple(agent_data["position"])
                energy = agent_data["energy"]
                is_tied = agent_data["is_tied_to_tracks"]

                # Find the same agent in the next turn
                next_agent_data = None
                for a in next_turn["agents"]:
                    if a["id"] == agent_id:
                        next_agent_data = a
                        break

                if next_agent_data is None:
                    continue

                # Calculate changes
                next_position = tuple(next_agent_data["position"])
                next_energy = next_agent_data["energy"]
                next_is_tied = next_agent_data["is_tied_to_tracks"]

                energy_change = next_energy - energy
                position_change = (
                    next_position[0] - position[0],
                    next_position[1] - position[1],
                )
                tied_change = int(next_is_tied) - int(is_tied)

                # Get affinity changes
                affinity_changes = {}
                if i + 1 < len(record["turns"]):
                    current_affinities = current_turn["affinity_matrix"]
                    next_affinities = next_turn["affinity_matrix"]

                    for key in current_affinities:
                        if key in next_affinities:
                            agent1, agent2 = map(int, key.split("_"))
                            if agent1 == agent_id or agent2 == agent_id:
                                affinity_changes[key] = (
                                    next_affinities[key] - current_affinities[key]
                                )

                # Trolley position
                trolley_position = current_turn["trolley_position"]
                if trolley_position:
                    distance_to_trolley = abs(position[0] - trolley_position[0]) + abs(
                        position[1] - trolley_position[1]
                    )
                else:
                    distance_to_trolley = None

                # Save data
                row = {
                    "game_id": game_id,
                    "turn": current_turn["turn_number"],
                    "agent_id": agent_id,
                    "action": action,
                    "energy": energy,
                    "energy_change": energy_change,
                    "is_tied": is_tied,
                    "tied_change": tied_change,
                    "distance_to_trolley": distance_to_trolley,
                    "position_change_x": position_change[0],
                    "position_change_y": position_change[1],
                }

                # Add affinity changes
                for key, change in affinity_changes.items():
                    row[f"affinity_change_{key}"] = change

                action_data.append(row)

    return pd.DataFrame(action_data)


def analyze_social_interactions(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze social interactions between agents

    Args:
        records: List of game records

    Returns:
        DataFrame containing social interaction data
    """
    interaction_data = []

    for record in records:
        game_id = record["game_id"]

        # Get agent personality profiles
        agent_profiles = {
            int(agent_id): profile
            for agent_id, profile in record["agent_profiles"].items()
        }

        # Process each turn
        for turn_idx, turn in enumerate(record["turns"]):
            turn_number = turn["turn_number"]

            # Find co-located agents (potential interactions)
            position_to_agents = {}
            for agent_data in turn["agents"]:
                if agent_data["alive"]:
                    position = tuple(agent_data["position"])
                    if position not in position_to_agents:
                        position_to_agents[position] = []
                    position_to_agents[position].append(agent_data)

            # Process each location with multiple agents
            for position, agents_at_pos in position_to_agents.items():
                if len(agents_at_pos) > 1:
                    # Look at each pair of agents
                    for i, agent1 in enumerate(agents_at_pos):
                        for agent2 in agents_at_pos[i + 1 :]:
                            agent1_id = agent1["id"]
                            agent2_id = agent2["id"]

                            # Get their affinity values
                            affinity_key1 = f"{agent1_id}_{agent2_id}"
                            affinity_key2 = f"{agent2_id}_{agent1_id}"

                            affinity1to2 = turn["affinity_matrix"].get(affinity_key1, 0)
                            affinity2to1 = turn["affinity_matrix"].get(affinity_key2, 0)

                            # Check if there was an interaction (based on last action)
                            interaction = None
                            if "last_action" in agent1:
                                action = agent1["last_action"]
                                if action in [
                                    "GIFT_ENERGY",
                                    "TIE_TO_TRACKS",
                                    "UNTIE_PRISONER",
                                ]:
                                    interaction = action

                            if "last_action" in agent2:
                                action = agent2["last_action"]
                                if action in [
                                    "GIFT_ENERGY",
                                    "TIE_TO_TRACKS",
                                    "UNTIE_PRISONER",
                                ]:
                                    interaction = action

                            # Get personality data if available
                            agent1_altruism = (
                                agent_profiles[agent1_id].get("altruism", None)
                                if agent1_id in agent_profiles
                                else None
                            )
                            agent1_aggression = (
                                agent_profiles[agent1_id].get("aggression", None)
                                if agent1_id in agent_profiles
                                else None
                            )
                            agent2_altruism = (
                                agent_profiles[agent2_id].get("altruism", None)
                                if agent2_id in agent_profiles
                                else None
                            )
                            agent2_aggression = (
                                agent_profiles[agent2_id].get("aggression", None)
                                if agent2_id in agent_profiles
                                else None
                            )

                            # Record interaction
                            row = {
                                "game_id": game_id,
                                "turn": turn_number,
                                "agent1_id": agent1_id,
                                "agent2_id": agent2_id,
                                "agent1_energy": agent1["energy"],
                                "agent2_energy": agent2["energy"],
                                "affinity1to2": affinity1to2,
                                "affinity2to1": affinity2to1,
                                "agent1_altruism": agent1_altruism,
                                "agent1_aggression": agent1_aggression,
                                "agent2_altruism": agent2_altruism,
                                "agent2_aggression": agent2_aggression,
                                "interaction": interaction,
                                "position_x": position[0],
                                "position_y": position[1],
                                "agent1_tied": agent1["is_tied_to_tracks"],
                                "agent2_tied": agent2["is_tied_to_tracks"],
                            }

                            interaction_data.append(row)

    return pd.DataFrame(interaction_data)


def analyze_trolley_dilemmas(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze trolley dilemma situations specifically

    Args:
        records: List of game records

    Returns:
        DataFrame containing trolley dilemma data
    """
    dilemma_data = []

    for record in records:
        game_id = record["game_id"]
        switch_position = record["game_config"]["switch_position"]

        # Process each turn where the trolley exists
        for turn_idx, turn in enumerate(record["turns"]):
            trolley_position = turn["trolley_position"]
            if not trolley_position:
                continue

            turn_number = turn["turn_number"]
            switch_state = turn["switch_state"]

            # Find agents on each branch
            agents_upper_branch = []
            agents_lower_branch = []
            agents_at_switch = []

            switch_x, switch_y = switch_position
            for agent_data in turn["agents"]:
                if not agent_data["alive"]:
                    continue

                agent_x, agent_y = agent_data["position"]

                # Agent at switch
                if agent_x == switch_x and agent_y == switch_y:
                    agents_at_switch.append(agent_data)
                # Agent on upper branch after switch
                elif agent_y == switch_y - 1 and agent_x > switch_x:
                    agents_upper_branch.append(agent_data)
                # Agent on lower branch after switch
                elif agent_y == switch_y + 1 and agent_x > switch_x:
                    agents_lower_branch.append(agent_data)

            # If trolley is approaching the switch and there are agents on the tracks
            if trolley_position[0] < switch_x and (
                agents_upper_branch or agents_lower_branch or agents_at_switch
            ):

                # This is a dilemma situation - record data about it
                row = {
                    "game_id": game_id,
                    "turn": turn_number,
                    "trolley_x": trolley_position[0],
                    "trolley_y": trolley_position[1],
                    "switch_state": switch_state,
                    "agents_on_upper": len(agents_upper_branch),
                    "agents_on_lower": len(agents_lower_branch),
                    "agents_at_switch": len(agents_at_switch),
                }

                # Add data about switch toggling
                if turn_idx > 0 and turn_idx < len(record["turns"]) - 1:
                    prev_turn = record["turns"][turn_idx - 1]
                    next_turn = record["turns"][turn_idx + 1]

                    # Was switch toggled in this turn?
                    switch_toggled = prev_turn["switch_state"] != switch_state
                    row["switch_toggled"] = int(switch_toggled)

                    # Will switch be toggled next turn?
                    will_toggle = switch_state != next_turn["switch_state"]
                    row["switch_will_toggle"] = int(will_toggle)

                    # Find agents at switch who might toggle it
                    for agent_data in agents_at_switch:
                        if (
                            "last_action" in agent_data
                            and agent_data["last_action"] == "CHANGE_SWITCH_STATE"
                        ):
                            row["agent_toggled_id"] = agent_data["id"]
                            if str(agent_data["id"]) in record["agent_profiles"]:
                                profile = record["agent_profiles"][
                                    str(agent_data["id"])
                                ]
                                row["toggler_altruism"] = profile.get("altruism", None)
                                row["toggler_risk_aversion"] = profile.get(
                                    "risk_aversion", None
                                )
                                row["toggler_aggression"] = profile.get(
                                    "aggression", None
                                )

                dilemma_data.append(row)

    return pd.DataFrame(dilemma_data)


def generate_training_data(
    records: List[Dict[str, Any]], output_file: str = "training_data.csv"
) -> None:
    """
    Generate training data for machine learning models

    Args:
        records: List of game records
        output_file: File to save the training data
    """
    # Create a DataFrame to hold the training examples
    training_examples = []

    for record in records:
        # Process each turn except the last one
        for i in range(len(record["turns"]) - 1):
            current_turn = record["turns"][i]
            next_turn = record["turns"][i + 1]

            # Process each agent
            for agent_data in current_turn["agents"]:
                if not agent_data["alive"] or "last_action" not in agent_data:
                    continue

                agent_id = agent_id = agent_data["id"]
                action = agent_data["last_action"]

                # Only include actions we want to learn
                if action not in [
                    "MOVE_UP",
                    "MOVE_DOWN",
                    "MOVE_LEFT",
                    "MOVE_RIGHT",
                    "WAIT",
                    "CHANGE_SWITCH_STATE",
                    "GIFT_ENERGY",
                    "TIE_TO_TRACKS",
                    "UNTIE_PRISONER",
                ]:
                    continue

                # Extract features that were available at the time of decision
                features = {}

                # Agent's current state
                features["agent_energy"] = agent_data["energy"]
                features["agent_tied"] = int(agent_data["is_tied_to_tracks"])
                features["agent_x"], features["agent_y"] = agent_data["position"]

                # Get agent personality traits
                if str(agent_id) in record["agent_profiles"]:
                    profile = record["agent_profiles"][str(agent_id)]
                    features["agent_altruism"] = profile.get("altruism", 0.5)
                    features["agent_risk_aversion"] = profile.get("risk_aversion", 0.5)
                    features["agent_aggression"] = profile.get("aggression", 0.5)
                else:
                    features["agent_altruism"] = 0.5
                    features["agent_risk_aversion"] = 0.5
                    features["agent_aggression"] = 0.5

                # Trolley information
                if current_turn["trolley_position"]:
                    features["trolley_x"], features["trolley_y"] = current_turn[
                        "trolley_position"
                    ]
                    features["distance_to_trolley"] = abs(
                        features["agent_x"] - features["trolley_x"]
                    ) + abs(features["agent_y"] - features["trolley_y"])
                    features["trolley_present"] = 1
                else:
                    features["trolley_x"], features["trolley_y"] = -1, -1
                    features["distance_to_trolley"] = -1
                    features["trolley_present"] = 0

                # Switch information
                switch_x, switch_y = record["game_config"]["switch_position"]
                features["switch_x"], features["switch_y"] = switch_x, switch_y
                features["switch_state"] = int(current_turn["switch_state"])
                features["distance_to_switch"] = abs(
                    features["agent_x"] - switch_x
                ) + abs(features["agent_y"] - switch_y)
                features["is_at_switch"] = int(features["distance_to_switch"] == 0)

                # Track information - is agent on track?
                on_track = 0
                agent_x, agent_y = features["agent_x"], features["agent_y"]
                if (
                    (agent_y == switch_y and agent_x <= switch_x)
                    or (agent_y == switch_y - 1 and agent_x > switch_x)
                    or (agent_y == switch_y + 1 and agent_x > switch_x)
                ):
                    on_track = 1
                features["on_track"] = on_track

                # Food information
                has_food = 0
                food_positions = []
                # Food is at the top and bottom edges in alternating phases
                if current_turn["turn_number"] % 200 < 100:
                    # Food at top row
                    food_y = 0
                else:
                    # Food at bottom row
                    food_y = record["game_config"]["height"] - 1

                # We don't have explicit food positions in the record,
                # so we approximate based on the game logic
                features["food_edge_y"] = food_y
                features["distance_to_food_edge"] = abs(agent_y - food_y)

                # Nearby agents
                nearby_agents = []
                for other_agent in current_turn["agents"]:
                    if other_agent["id"] != agent_id and other_agent["alive"]:
                        other_x, other_y = other_agent["position"]
                        manhattan_dist = abs(agent_x - other_x) + abs(agent_y - other_y)
                        if manhattan_dist <= 3:  # Consider agents within distance 3
                            nearby_agents.append(
                                {
                                    "id": other_agent["id"],
                                    "distance": manhattan_dist,
                                    "energy": other_agent["energy"],
                                    "tied": int(other_agent["is_tied_to_tracks"]),
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
                    features["colocated_tied_agents"] = sum(
                        a["tied"] for a in colocated_agents
                    )
                else:
                    features["colocated_min_energy"] = -1
                    features["colocated_max_energy"] = -1
                    features["colocated_tied_agents"] = 0

                # Affinity information
                affinity_sums = 0
                affinity_count = 0

                for key, value in current_turn["affinity_matrix"].items():
                    id1, id2 = map(int, key.split("_"))
                    if id1 == agent_id:
                        affinity_sums += value
                        affinity_count += 1

                features["avg_affinity_to_others"] = affinity_sums / max(
                    affinity_count, 1
                )

                # Reward metrics (outcome of action)
                # This is what we're trying to predict/learn, but also useful for training
                next_agent_data = None
                for a in next_turn["agents"]:
                    if a["id"] == agent_id:
                        next_agent_data = a
                        break

                if next_agent_data:
                    # Changes in agent state
                    energy_change = next_agent_data["energy"] - agent_data["energy"]
                    score_change = next_agent_data["score"] - agent_data["score"]
                    tied_change = int(next_agent_data["is_tied_to_tracks"]) - int(
                        agent_data["is_tied_to_tracks"]
                    )

                    # Add the action and outcomes
                    example = {
                        "game_id": record["game_id"],
                        "turn": current_turn["turn_number"],
                        "agent_id": agent_id,
                        "action": action,
                        "energy_change": energy_change,
                        "score_change": score_change,
                        "tied_change": tied_change,
                        "survived": int(next_agent_data["alive"]),
                    }

                    # Add all the features
                    example.update(features)

                    training_examples.append(example)

    # Convert to DataFrame and save
    df = pd.DataFrame(training_examples)
    df.to_csv(output_file, index=False)
    print(f"Training data saved to {output_file}")


def create_visualizations(data_dir: str, output_dir: str = "analysis") -> None:
    """
    Create visualizations of game data analysis

    Args:
        data_dir: Directory containing game record JSON files
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load game records
    records = load_game_records(data_dir)
    if not records:
        print("No game records found.")
        return

    print(f"Loaded {len(records)} game records.")

    # Extract agent performance data
    agent_df = extract_agent_performance(records)
    action_df = analyze_action_outcomes(records)
    interaction_df = analyze_social_interactions(records)
    dilemma_df = analyze_trolley_dilemmas(records)

    # Save DataFrames
    agent_df.to_csv(f"{output_dir}/agent_performance.csv", index=False)
    action_df.to_csv(f"{output_dir}/action_outcomes.csv", index=False)
    interaction_df.to_csv(f"{output_dir}/social_interactions.csv", index=False)
    dilemma_df.to_csv(f"{output_dir}/trolley_dilemmas.csv", index=False)

    # Create visualizations

    # 1. Plot distribution of final scores by personality traits
    plt.figure(figsize=(10, 6))
    plt.scatter(
        agent_df["altruism"], agent_df["final_score"], alpha=0.5, label="Altruism"
    )
    plt.scatter(
        agent_df["risk_aversion"],
        agent_df["final_score"],
        alpha=0.5,
        label="Risk Aversion",
    )
    plt.scatter(
        agent_df["aggression"], agent_df["final_score"], alpha=0.5, label="Aggression"
    )
    plt.xlabel("Personality Trait Value")
    plt.ylabel("Final Score")
    plt.title("Agent Scores by Personality Traits")
    plt.legend()
    plt.savefig(f"{output_dir}/scores_by_traits.png")

    # 2. Plot survival rates by risk aversion
    plt.figure(figsize=(10, 6))
    agent_df["risk_bin"] = pd.cut(agent_df["risk_aversion"], bins=5)
    survival_by_risk = agent_df.groupby("risk_bin")["survival_turns"].mean()
    survival_by_risk.plot(kind="bar")
    plt.xlabel("Risk Aversion Level")
    plt.ylabel("Average Survival Turns")
    plt.title("Survival Turns by Risk Aversion")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/survival_by_risk.png")

    # 3. Plot action frequencies
    action_columns = [col for col in agent_df.columns if col.startswith("action_")]
    if action_columns:
        action_sums = agent_df[action_columns].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        ax = action_sums.plot(
            kind="bar"
        )  # Store the axis returned by the plot function
        ax.set_yscale("log")  # Set the y-axis to a logarithmic scale
        plt.xlabel("Action")
        plt.ylabel("Frequency (log scale)")  # Update ylabel to indicate log scale
        plt.title("Action Frequencies Across All Agents (Log Scale)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/action_frequencies.png")

    # 4. Plot trolley dilemma outcomes
    if not dilemma_df.empty:
        plt.figure(figsize=(10, 6))
        dilemma_counts = (
            dilemma_df.groupby(["agents_on_upper", "agents_on_lower", "switch_toggled"])
            .size()
            .unstack()
        )
        dilemma_counts.plot(kind="bar", stacked=True)
        plt.xlabel("Agents on Tracks (Upper, Lower)")
        plt.ylabel("Count")
        plt.title("Switch Toggle Decisions by Agent Distributions")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dilemma_decisions.png")

    # 5. Plot altruism vs. energy gifting
    gift_actions = interaction_df[interaction_df["interaction"] == "GIFT_ENERGY"]
    if not gift_actions.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(gift_actions["agent1_altruism"], gift_actions["agent1_energy"])
        plt.xlabel("Agent Altruism")
        plt.ylabel("Agent Energy When Gifting")
        plt.title("Energy Gifting by Altruism Level")
        plt.savefig(f"{output_dir}/gifting_by_altruism.png")

    # Generate training data
    generate_training_data(records, f"{output_dir}/training_data.csv")

    print(f"Analysis complete. Visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trolley problem game data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="game_data",
        help="Directory containing game record JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to analyze",
    )

    args = parser.parse_args()

    create_visualizations(args.data_dir, args.output_dir)
