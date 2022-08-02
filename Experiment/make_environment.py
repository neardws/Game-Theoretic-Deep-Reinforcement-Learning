import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")

from typing import Optional, List, Tuple
import numpy as np
from Environment.environment import vehicularNetworkEnv, init_distance_matrix_and_radio_coverage_matrix, define_size_of_spaces
from Environment.environmentConfig import vehicularNetworkEnvConfig
from Environment.dataStruct import vehicleList, timeSlots, taskList, edgeList
from Utilities.FileOperator import save_obj, init_file_name

def get_default_environment(
        flatten_space: Optional[bool] = False,
        occuiped: Optional[bool] = False,
        for_mad5pg: Optional[bool] = False,
    ) -> Tuple[timeSlots, taskList, vehicleList, edgeList, np.ndarray, np.ndarray, List[List[List[int]]], vehicularNetworkEnvConfig, vehicularNetworkEnv]:
    
    environment_config = vehicularNetworkEnvConfig(
        task_request_rate=0.35,
    )
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
        minimum_delay_thresholds=environment_config.task_minimum_delay_thresholds,
        maximum_delay_thresholds=environment_config.task_maximum_delay_thresholds,
        seed=environment_config.task_seed,
    )
    
    vehicle_list = vehicleList(
        edge_number=environment_config.edge_number,
        communication_range=environment_config.communication_range,
        vehicle_number=environment_config.vehicle_number,
        time_slots=time_slots,
        trajectories_file_name=environment_config.trajectories_file_name,
        slot_number=environment_config.time_slot_number,
        task_number=environment_config.task_number,
        task_request_rate=environment_config.task_request_rate,
        seeds=environment_config.vehicle_seeds,
    )
    
    # print("len(vehicle_list): ", len(vehicle_list.get_vehicle_list()))
    # print("vehicle_number: ", environment_config.vehicle_number)
    
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
    
    distance_matrix, channel_condition_matrix, vehicle_index_within_edges, vehicle_observed_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list)
    
    environment_config.vehicle_number_within_edges = int(environment_config.vehicle_number / environment_config.edge_number)
    environment_config.action_size, environment_config.observation_size, environment_config.reward_size, \
            environment_config.critic_network_action_size = define_size_of_spaces(vehicle_number_within_edges=environment_config.vehicle_number_within_edges, edge_number=environment_config.edge_number, task_assigned_number=environment_config.task_assigned_number)
    
    print("environment_config.action_size: ", environment_config.action_size)
    print("environment_config.observation_size: ", environment_config.observation_size)
    print("environment_config.reward_size: ", environment_config.reward_size)
    print("environment_config.critic_network_action_size: ", environment_config.critic_network_action_size)
    
    environment = vehicularNetworkEnv(
        envConfig = environment_config,
        time_slots = time_slots,
        task_list = task_list,
        vehicle_list = vehicle_list,
        edge_list = edge_list,
        distance_matrix = distance_matrix, 
        channel_condition_matrix = channel_condition_matrix, 
        vehicle_index_within_edges = vehicle_index_within_edges,
        vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
        flatten_space = flatten_space,
        occuiped = occuiped,
        for_mad5pg = for_mad5pg, 
    )
    
    file_name = init_file_name()
    save_obj(environment, file_name["init_environment_name"])
    

if __name__ == "__main__":
    get_default_environment(for_mad5pg=True)