import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Environment.environment import make_environment_spec
from Agents.DDPG.agent import DDPG
from acme import types
from typing import Dict, Sequence
import sonnet as snt
from acme.tf import networks
import numpy as np
import tensorflow as tf
from Experiment.make_environment import get_default_environment

def make_networks(
    action_spec: types.NestedSpec,
    policy_layer_sizes: Sequence[int] = (128, 128),
    critic_layer_sizes: Sequence[int] = (256, 128),
    ) -> Dict[str, snt.Module]:
    """Creates networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)
    policy_layer_sizes = list(policy_layer_sizes) + [num_dimensions]
    critic_layer_sizes = list(critic_layer_sizes) + [1]

    policy_network = snt.Sequential(
        [networks.LayerNormMLP(policy_layer_sizes), tf.tanh])
    # The multiplexer concatenates the (maybe transformed) observations/actions.
    critic_network = networks.CriticMultiplexer(
        critic_network=networks.LayerNormMLP(critic_layer_sizes))

    return {
        'policy': policy_network,
        'critic': critic_network,
    }

def main(_):
    
    __, __, __, __, __, __, __, __, environment = get_default_environment(flatten_space=True)

    
    env_spec = make_environment_spec(environment)
    
    agent_networks = make_networks(env_spec.actions)
    
    agent = DDPG(
        environment_spec=env_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
    )

    
    # Create the environment loop used for training.
    train_loop = EnvironmentLoop(environment, agent, label='train_loop')

    train_loop.run(num_episodes=5000)