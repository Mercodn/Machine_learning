"""
Reinforcement Learning Model
Simple multi-armed bandit reward and punishment environment.
"""

import random

BANDIT_PROBABILITIES = [0.2, 0.5, 0.8]
BANDIT_NAMES = ["Arm 1", "Arm 2", "Arm 3"]


def get_bandit_info():
    return {
        'n_arms': len(BANDIT_PROBABILITIES),
        'arm_names': BANDIT_NAMES,
        'probabilities': BANDIT_PROBABILITIES,
        'description': 'A simple reinforcement learning environment where the agent chooses an action and receives reward or punishment.'
    }


def simulate_action(arm_index):
    probability = BANDIT_PROBABILITIES[arm_index]
    reward = 1 if random.random() < probability else -1
    return reward, probability


def update_value_estimates(action_counts, action_values, arm_index, reward):
    updated_counts = [int(x) for x in action_counts]
    updated_values = [float(x) for x in action_values]
    updated_counts[arm_index] += 1
    old_value = updated_values[arm_index]
    updated_values[arm_index] = old_value + (reward - old_value) / updated_counts[arm_index]
    return updated_counts, updated_values
