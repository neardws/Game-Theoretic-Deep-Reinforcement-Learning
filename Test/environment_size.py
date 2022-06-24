
import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")

from Environment.environment import vehicularNetworkEnv, make_environment_spec, init_distance_matrix_and_radio_coverage_matrix, define_size_of_spaces, get_maximum_vehicle_number
from Environment.environmentConfig import vehicularNetworkEnvConfig
from Environment.dataStruct import get_vehicle_number, vehicleList, timeSlots, taskList, edgeList

if __name__ == "__main__":
    
    for _ in range(1):
        print("_________________________________________________")
        print("index:", _)
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
        
        distance_matrix, channel_condition_matrix, vehicle_number_within_edges, \
                vehicle_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list)
        
        print("vehicle_index_within_edges: ", len(vehicle_index_within_edges))
        print("vehicle_index_within_edges_1: ", len(vehicle_number_within_edges[0]))
        
        for edge_index in range(environment_config.edge_number):
            for time_index in range(environment_config.time_slot_number):
                vehicle_number = vehicle_number_within_edges[edge_index][time_index]
                vehicle_index_number = len(vehicle_index_within_edges[edge_index][time_index])
                if vehicle_number != vehicle_index_number:
                    error_info = "edge_index: ", edge_index + " time_index: ", edge_index + " vehicle_number: ", vehicle_number + " vehicle_index_number: ", vehicle_index_number + " vehicle_index: ", vehicle_index_within_edges[edge_index][time_index]                
                    raise ValueError(error_info)
        
        environment_config.maximum_vehicle_number_within_edges = int(get_maximum_vehicle_number(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list))
        environment_config.action_size, environment_config.observation_size, environment_config.reward_size, \
                environment_config.critic_network_action_size = define_size_of_spaces(maximum_vehicle_number_within_edges=environment_config.maximum_vehicle_number_within_edges, edge_number=environment_config.edge_number)
        
        print("vehicle_number:", environment_config.vehicle_number)
        environment = vehicularNetworkEnv(
            envConfig = environment_config,
            time_slots = time_slots,
            task_list = task_list,
            vehicle_list = vehicle_list,
            edge_list = edge_list,
            distance_matrix = distance_matrix, 
            channel_condition_matrix = channel_condition_matrix, 
            vehicle_number_within_edges = vehicle_number_within_edges,
            vehicle_index_within_edges = vehicle_index_within_edges,    
        )
        
        print("maximum_vehicle_number: ", environment_config.maximum_vehicle_number_within_edges)
        spec = make_environment_spec(environment)
        print("spec.edge_observations: ", spec.edge_observations)
        print("spec.edge_actions: ", spec.edge_actions)
