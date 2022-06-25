import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Environment.environment import vehicularNetworkEnv, make_environment_spec, init_distance_matrix_and_radio_coverage_matrix, define_size_of_spaces, get_maximum_vehicle_number
from Environment.environmentConfig import vehicularNetworkEnvConfig
from Environment.dataStruct import get_vehicle_number, vehicleList, timeSlots, taskList, edgeList
from Agents.MPO.agent import MPO
from acme import types
from typing import Dict, Sequence
import sonnet as snt
from acme.tf import networks
import numpy as np
import tensorflow as tf

def make_networks(
    action_spec: types.NestedSpec,
    policy_layer_sizes: Sequence[int] = (128, 128),
    critic_layer_sizes: Sequence[int] = (256, 256),
    ) -> Dict[str, snt.Module]:
    """Creates networks used by the agent."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)
    critic_layer_sizes = list(critic_layer_sizes) + [1]

    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes),
        networks.MultivariateNormalDiagHead(num_dimensions)
    ])
    critic_network = networks.CriticMultiplexer(
        critic_network=networks.LayerNormMLP(critic_layer_sizes))
    
    return {
        'policy': policy_network,
        'critic': critic_network,
    }

def main(_):
    
    environment_config = vehicularNetworkEnvConfig()
    environment_config.vehicle_number = int(get_vehicle_number(environment_config.trajectories_file_name) * environment_config.vehicle_number_rate) 
    environment_config.vehicle_seeds += [i for i in range(environment_config.vehicle_number)]
    
    time_slots= timeSlots(
        start=environment_config.time_slot_start,
        end=environment_config.time_slot_end,
        slot_length=environment_config.time_slot_length,
    )
    
    task_list = taskList(
        tasks_number=environment_config.task_number,
        minimum_data_size=environment_config.task_minimum_data_size,
        maximum_data_size=environment_config.task_maximum_data_size,
        minimum_computation_cycles=environment_config.task_minimum_computation_cycles,
        maximum_computation_cycles=environment_config.task_maximum_computation_cycles,
        seed=environment_config.task_seed,
    )
    
    vehicle_list = vehicleList(
        vehicle_number=environment_config.vehicle_number,
        time_slots=time_slots,
        trajectories_file_name=environment_config.trajectories_file_name,
        slot_number=environment_config.time_slot_number,
        task_number=environment_config.task_number,
        task_request_rate=environment_config.task_request_rate,
        seeds=environment_config.vehicle_seeds,
    )
    
    edge_list = edgeList(
        edge_number=environment_config.edge_number,
        power=environment_config.edge_power,
        bandwidth=environment_config.edge_bandwidth,
        minimum_computing_cycles=environment_config.edge_minimum_computing_cycles,
        maximum_computing_cycles=environment_config.edge_maximum_computing_cycles,
        communication_range=environment_config.communication_range,
        edge_xs=[500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500],
        edge_ys=[2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500],
        seed=environment_config.edge_seed,
    )
    
    distance_matrix, channel_condition_matrix, vehicle_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list)
        
    environment_config.maximum_vehicle_number_within_edges = int(get_maximum_vehicle_number(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list))
    environment_config.action_size, environment_config.observation_size, environment_config.reward_size, \
            environment_config.critic_network_action_size = define_size_of_spaces(maximum_vehicle_number_within_edges=environment_config.maximum_vehicle_number_within_edges, edge_number=environment_config.edge_number)
    
    environment = vehicularNetworkEnv(
        envConfig = environment_config,
        time_slots = time_slots,
        task_list = task_list,
        vehicle_list = vehicle_list,
        edge_list = edge_list,
        distance_matrix = distance_matrix, 
        channel_condition_matrix = channel_condition_matrix, 
        vehicle_index_within_edges = vehicle_index_within_edges,
        flatten_space=True,
    )
    
    env_spec = make_environment_spec(environment)
    
    agent_networks = make_networks(env_spec.actions)
    
    agent = MPO(
        environment_spec=env_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
    )

    
    # Create the environment loop used for training.
    train_loop = EnvironmentLoop(environment, agent, label='train_loop')

    train_loop.run(num_episodes=5000)