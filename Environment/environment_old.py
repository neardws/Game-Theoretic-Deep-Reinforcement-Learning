
"""Vehicular Network Environments."""
import time
import dm_env
from dm_env import specs
from acme.types import NestedSpec
import numpy as np
from typing import List, Tuple, NamedTuple, Optional
import Environment.environmentConfig as env_config
from Environment.dataStruct import timeSlots, taskList, edgeList, vehicleList
from Environment.utilities import compute_channel_gain, generate_complex_normal_distribution, compute_transmission_rate, compute_SINR, cover_mW_to_W
np.set_printoptions(threshold=np.inf)
from Log.logger import myapp

class vehicularNetworkEnv(dm_env.Environment):
    """Vehicular Network Environment built on the dm_env framework."""
    
    def __init__(
        self, 
        envConfig: Optional[env_config.vehicularNetworkEnvConfig] = None,
        time_slots: Optional[timeSlots] = None,
        task_list: Optional[taskList] = None,
        vehicle_list: Optional[vehicleList] = None,
        edge_list: Optional[edgeList] = None,
        distance_matrix: Optional[np.ndarray] = None, 
        channel_condition_matrix: Optional[np.ndarray] = None, 
        vehicle_index_within_edges: Optional[List[List[List[int]]]] = None,
        vehicle_observed_index_within_edges: Optional[List[List[List[int]]]] = None,
        flatten_space: Optional[bool] = True,
        occuiped: Optional[bool] = False,
        for_mad5pg: Optional[bool] = False,
    ) -> None:
        """Initialize the environment."""
        if envConfig is None:
            self._config = env_config.vehicularNetworkEnvConfig()
            self._config.vehicle_seeds += [i for i in range(self._config.vehicle_number)]
            self._config.vehicle_number_within_edges = int(self._config.vehicle_number / self._config.edge_number)
            self._config.action_size, self._config.observation_size, self._config.reward_size, \
                self._config.critic_network_action_size = define_size_of_spaces(self._config.vehicle_number_within_edges, self._config.edge_number, self._config.task_assigned_number)
        else:
            self._config = envConfig
        
        if distance_matrix is None:
            self._distance_matrix, self._channel_condition_matrix, self._vehicle_index_within_edges, self._vehicle_observed_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(
                env_config=self._config,
                vehicle_list=vehicle_list,
                edge_list=edge_list,
            )
        else:
            self._distance_matrix = distance_matrix
            self._channel_condition_matrix = channel_condition_matrix
            self._vehicle_index_within_edges = vehicle_index_within_edges
            self._vehicle_observed_index_within_edges = vehicle_observed_index_within_edges
        
        if time_slots is None:
            self._time_slots: timeSlots = timeSlots(
                start=self._config.time_slot_start,
                end=self._config.time_slot_end,
                slot_length=self._config.time_slot_length,
            )
        else:
            self._time_slots = time_slots
        if task_list is None:
            self._task_list: taskList = taskList(
                tasks_number=self._config.task_number,
                minimum_data_size=self._config.task_minimum_data_size,
                maximum_data_size=self._config.task_maximum_data_size,
                minimum_computation_cycles=self._config.task_minimum_computation_cycles,
                maximum_computation_cycles=self._config.task_maximum_computation_cycles,
                minimum_delay_thresholds=self._config.task_minimum_delay_thresholds,
                maximum_delay_thresholds=self._config.task_maximum_delay_thresholds,
                seed=self._config.task_seed,
            )
        else:
            self._task_list = task_list
        if vehicle_list is None:
            self._vehicle_list: vehicleList = vehicleList(
                vehicle_number=self._config.vehicle_number,
                time_slots=self._time_slots,
                trajectories_file_name=self._config.trajectories_file_name,
                slot_number=self._config.time_slot_number,
                task_number=self._config.task_number,
                task_request_rate=self._config.task_request_rate,
                seeds=self._config.vehicle_seeds,
            )
        else:
            self._vehicle_list = vehicle_list
        if edge_list is None:
            self._edge_list: edgeList = edgeList(
                edge_number=self._config.edge_number,
                power=self._config.edge_power,
                bandwidth=self._config.edge_bandwidth,
                minimum_computing_cycles=self._config.edge_minimum_computing_cycles,
                maximum_computing_cycles=self._config.edge_maximum_computing_cycles,
                communication_range=self._config.communication_range,
                edge_xs=[500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500],
                edge_ys=[2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500],
                seed=self._config.edge_seed,
            )
        else:
            self._edge_list = edge_list
        
        self._reward: np.ndarray = np.zeros(self._config.reward_size)
        
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        self._reset_next_step: bool = True
        self._flatten_space: bool = flatten_space
        self._occuiped: bool = occuiped
        self._for_mad5pg: bool = for_mad5pg
        
    def reset(self) -> dm_env.TimeStep:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        Returns the first `TimeStep` of a new episode.
        """
        self._time_slots.reset()
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        self._reset_next_step = False
        
        return dm_env.restart(observation=self._observation())

    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """        
        if self._reset_next_step:
            return self.reset()
        time_start = time.time()
        self._reward, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_required_number = self.compute_reward(action)
        # print("compute_reward time taken: ", time.time() - time_start)
        
        time_start = time.time()
        observation = self._observation()
        # print("observation time taken: ", time.time() - time_start)
        # check for termination
        if self._time_slots.is_end():
            self._reset_next_step = True
            return dm_env.termination(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_required_number
        self._time_slots.add_time()
        return dm_env.transition(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_required_number

    def compute_reward(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, float, float]:
        
        time_start = time.time()
        actions = np.array(action)
        
        punished_time = 30
        # print("actions1: ", actions)
        if self._flatten_space:
            actions = np.reshape(np.array(actions), newshape=(self._config.edge_number, self._config.action_size))

        # print("actions2: ", actions)
        vehicle_SINR = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_execution_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_wired_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_task_assignment = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_computation_resources = np.zeros((self._config.vehicle_number, self._config.edge_number))    
        
        reward_part_1_time = time.time() - time_start
        
        cumulative_reward = 0    
        successful_serviced_number = 0
        task_required_number = 0
        
        time_start = time.time()
        for edge_index in range(self._config.edge_number):
            try:
                vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            except IndexError:
                raise IndexError("edge_index: ", edge_index, "self._time_slots.now(): ", self._time_slots.now())
            tasks_number_within_edge = len(vehicle_index_within_edge)
            vehicle_observed_index_within_edge = self._vehicle_observed_index_within_edges[edge_index][self._time_slots.now()]
            vehicle_number_within_edge = len(vehicle_observed_index_within_edge)
            
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            
            transmission_power_allocation = np.array(actions[edge_index, : int(self._config.vehicle_number_within_edges)])
            input_array = transmission_power_allocation
            power_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
                    
            sorted_vehicle_index_within_edge = vehicle_index_within_edge
            """sorted by the channel condition"""
            # sorted_vehicle_index_within_edge = sorted(sorted_vehicle_index_within_edge, key=lambda x: self._channel_condition_matrix[x][edge_index][self._time_slots.now()])
            # sorted_vehicle_index_within_edge = sorted(sorted_vehicle_index_within_edge, key=lambda x: self._channel_condition_matrix[x][edge_index][self._time_slots.now()], reverse=True)
            sorted_power_allocation = power_allocation
            
            edge_power = the_edge.get_power()
            edge_occupied_power = self._occupied_power[edge_index][self._time_slots.now()]
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = sorted_vehicle_index_within_edge[i]
                try:
                    transmission_power = sorted_power_allocation[i]
                    if self._occuiped:
                        if edge_power - edge_occupied_power <= 0:
                            vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
                        else:
                            vehicle_edge_transmission_power[vehicle_index][edge_index] = transmission_power * (edge_power - edge_occupied_power)
                    else:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = transmission_power * edge_power
                except IndexError:
                    pass
                        
            task_assignment = np.array(actions[edge_index, int(self._config.vehicle_number_within_edges) : ])
            task_assignment = np.reshape(task_assignment, newshape=(self._config.vehicle_number_within_edges, self._config.edge_number))
            
            for i in range(int(vehicle_number_within_edge)):
                try:
                    processing_edge_index = int(task_assignment[i, :].argmax())
                    
                    vehicle_index = vehicle_observed_index_within_edge[i]                    
                    if self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now()) != -1:
                        
                        vehicle_edge_task_assignment[vehicle_index][processing_edge_index] = 1
                    
                        if processing_edge_index != edge_index:
                            task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                            data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                            wired_transmission_time = data_size / self._config.wired_transmission_rate * self._config.wired_transmission_discount * \
                                    the_edge.get_edge_location().get_distance(self._edge_list.get_edge_by_index(processing_edge_index).get_edge_location())
                            for e in range(self._config.edge_number + 1):
                                vehicle_wired_transmission_time[vehicle_index, e] = wired_transmission_time
                except IndexError:
                    pass
            
            # assigned_at_local = 0.8
            # small_number = 0.000000001
            # for i in range(int(tasks_number_within_edge)):
            #     if task_assignment[i] <= assigned_at_local:
            #         processing_edge_index = edge_index
            #     else:
            #         difference = task_assignment[i] - assigned_at_local
            #         if difference < 0:
            #             difference = small_number
            #         if difference > assigned_at_local:
            #             difference = assigned_at_local - small_number
            #         processing_edge_index = int(np.floor(difference * (1 / (1 - assigned_at_local))  / (1 / (self._config.edge_number - 1))))
            #         # print(f"difference: {difference}, (1 / (1 - assigned_at_local)): {(1 / (1 - assigned_at_local))},  difference * (1 / (1 - assigned_at_local)): {difference * (1 / (1 - assigned_at_local))}, (1 / (self._config.edge_number - 1)): {(1 / (self._config.edge_number - 1))}, difference * (1 / (1 - assigned_at_local))  / (1 / (self._config.edge_number - 1)): {difference * (1 / (1 - assigned_at_local))  / (1 / (self._config.edge_number - 1))}, np.floor(difference * (1 / (1 - assigned_at_local))  / (1 / (self._config.edge_number - 1))): {np.floor(difference * (1 / (1 - assigned_at_local))  / (1 / (self._config.edge_number - 1)))} ")
            #         # print("processing_edge_index: ", processing_edge_index)
            #         if processing_edge_index < edge_index:
            #             processing_edge_index = processing_edge_index
            #         if processing_edge_index >= edge_index:
            #             processing_edge_index += 1
            #             if processing_edge_index == self._config.edge_number:
            #                 processing_edge_index -= 1
            #     vehicle_index = vehicle_index_within_edge[i]
            #     vehicle_edge_task_assignment[vehicle_index][processing_edge_index] = 1
            
            #     if processing_edge_index != edge_index:
            #         task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
            #         data_size = self._task_list.get_task_by_index(task_index).get_data_size()
            #         wired_transmission_time = data_size / self._config.wired_transmission_rate * self._config.wired_transmission_discount * \
            #                 the_edge.get_edge_location().get_distance(self._edge_list.get_edge_by_index(processing_edge_index).get_edge_location())
            #         for e in range(self._config.edge_number + 1):
            #             vehicle_wired_transmission_time[vehicle_index, e] = wired_transmission_time
        
        reward_part_2_time = time.time() - time_start
        
        time_start = time.time()
        
        for edge_index in range(self._config.edge_number):

            edge_computing_speed = self._edge_list.get_edge_by_index(edge_index).get_computing_speed()
            edge_occupied_computing_speed = self._occupied_computing_resources[edge_index][self._time_slots.now()]
            # computation_resource_allocation = np.array(actions[edge_index, int(self._config.vehicle_number_within_edges * 2): ] )
            
            task_sum = int(np.sum(vehicle_edge_task_assignment[:, edge_index]))
            
            # new_computation_resource_allocation = np.zeros(int(task_sum))
            task_vehicle_index = np.where(vehicle_edge_task_assignment[:, edge_index] == 1)[0]
            # for i in range(int(task_sum)):
            #     vehicle_index = task_vehicle_index[i]
            #     new_computation_resource_allocation[i] = computation_resource_allocation[vehicle_index]
            
            # input_array = new_computation_resource_allocation
            # computation_resource_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
            
            # print("task_sum: ", task_sum)    
                    
            # if task_sum <= self._config.vehicle_number_within_edges * self._config.task_assigned_number:
            #     input_array = computation_resource_allocation[: task_sum]
            #     computation_resource_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
            # else:
                
            #     task_sum = self._config.vehicle_number_within_edges * self._config.task_assigned_number
            #     input_array = computation_resource_allocation
            #     computation_resource_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
            #     for i in range(task_sum, int(np.sum(vehicle_edge_task_assignment[:, edge_index]))):
            #         vehicle_index = task_vehicle_index[i]
            #         vehicle_execution_time[vehicle_index, -1] = punished_time
            
            for i in range(task_sum):
                vehicle_index = task_vehicle_index[i]
                if self._occuiped:
                    if edge_computing_speed - edge_occupied_computing_speed <= 0:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = 1 / task_sum * (edge_computing_speed - edge_occupied_computing_speed)
                else:
                    vehicle_edge_computation_resources[vehicle_index][edge_index] = 1 / task_sum * edge_computing_speed
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                if vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                    if float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index]) < punished_time:
                        vehicle_execution_time[vehicle_index, -1] = float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index])
                    else:
                        vehicle_execution_time[vehicle_index, -1] = punished_time
                else:
                    vehicle_execution_time[vehicle_index, -1] = punished_time
                    
                for e in range(self._config.edge_number):  # e is the edge node which do nothing
                    if e == edge_index:
                        vehicle_execution_time[vehicle_index, e] = punished_time
                    else:
                        vehicle_execution_time[vehicle_index, e] = vehicle_execution_time[vehicle_index, -1]
                        
                if vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                    occupied_time = int(np.floor(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index]))
                    if self._occuiped and occupied_time > 0:
                        start_time = int(self._time_slots.now() + 1)
                        end_time = int(self._time_slots.now() + occupied_time + 1)
                        if end_time < self._config.time_slot_number:
                            for i in range(start_time, end_time):
                                self._occupied_computing_resources[edge_index][i] += vehicle_edge_computation_resources[vehicle_index][edge_index]
                        else:
                            for i in range(start_time, int(self._config.time_slot_number)):
                                self._occupied_computing_resources[edge_index][i] += vehicle_edge_computation_resources[vehicle_index][edge_index]
        
        reward_part_3_time = time.time() - time_start
        
        time_start = time.time()
        
        reward_part_7_time = 0
        reward_part_8_time = 0
        reward_part_9_time = 0
        """Compute the inference"""
        for edge_index in range(self._config.edge_number):
            
            time_start = time.time()
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            
            edge_inter_interference = np.zeros((self._config.edge_number))
            
            for other_edge_index in range(self._config.edge_number):
                if other_edge_index != edge_index:
                    vehicle_index_within_other_edge = self._vehicle_index_within_edges[other_edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_other_edge:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        inter_interference = np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][other_edge_index])
                        edge_inter_interference[other_edge_index] += inter_interference
            reward_part_9_time += time.time() - time_start
            
            if vehicle_index_within_edge != []:
                for vehicle_index in vehicle_index_within_edge:
                    
                    time_start = time.time()
                    for e in range(self._config.edge_number):
                        vehicle_inter_edge_inference[vehicle_index, -1] += edge_inter_interference[e]
                        for other_edge_index in range(self._config.edge_number):
                            if e == other_edge_index:
                                vehicle_inter_edge_inference[vehicle_index, other_edge_index] += 0
                            else:
                                vehicle_inter_edge_inference[vehicle_index, other_edge_index] += edge_inter_interference[e]
                    
                    reward_part_7_time += time.time() - time_start
                    
                    time_start = time.time()
                    channel_condition = self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_edge:
                        if other_vehicle_index != vehicle_index:
                            other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                            if other_channel_condition < channel_condition:
                                vehicle_intar_edge_inference[vehicle_index, -1] += np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][edge_index])
                    for e in range(self._config.edge_number):
                        if e == edge_index:
                            vehicle_intar_edge_inference[vehicle_index, e] = 0
                        else:
                            vehicle_intar_edge_inference[vehicle_index, e] = vehicle_intar_edge_inference[vehicle_index, -1]
                    reward_part_8_time += time.time() - time_start
        
        reward_part_4_time = time.time() - time_start
        
        time_start = time.time()
        
        """Compute the SINR and transimission time"""
        for edge_index in range(self._config.edge_number):
            if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
                for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                    
                    for e in range(self._config.edge_number):
                        if e == edge_index:
                            vehicle_transmission_time[vehicle_index, e] = punished_time
                        else:
                            vehicle_SINR[vehicle_index, e] = compute_SINR(
                                white_gaussian_noise=self._config.white_gaussian_noise, 
                                channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                                transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                                intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][e],
                                inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][e],)
                            transmission_rate = compute_transmission_rate(
                                SINR=vehicle_SINR[vehicle_index, e], 
                                bandwidth=self._config.edge_bandwidth)
                            if transmission_rate != 0:
                                if float(data_size / transmission_rate) < punished_time:
                                    vehicle_transmission_time[vehicle_index, e] = float(data_size / transmission_rate)
                                else:
                                    vehicle_transmission_time[vehicle_index, e] = punished_time
                            else:
                                vehicle_transmission_time[vehicle_index, e] = punished_time
                    
                    vehicle_SINR[vehicle_index, -1] = compute_SINR(
                        white_gaussian_noise=self._config.white_gaussian_noise, 
                        channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                        transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                        intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][-1],
                        inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][-1],)
                    
                    transmission_rate = compute_transmission_rate(
                        SINR=vehicle_SINR[vehicle_index, -1], 
                        bandwidth=self._config.edge_bandwidth)
                    
                    
                    if transmission_rate != 0:
                        if float(data_size / transmission_rate) < punished_time:
                            vehicle_transmission_time[vehicle_index, -1] = float(data_size / transmission_rate)
                        else:
                            vehicle_transmission_time[vehicle_index, -1] = punished_time
                    else:
                        vehicle_transmission_time[vehicle_index, -1] = punished_time
                        
                    # print("edge_index: {}, vehicle_index: {}, SINR: {} transmission_rate: {}, data_size: {}, transmission_time: {}".format(edge_index, vehicle_index, vehicle_SINR[vehicle_index, -1], transmission_rate, data_size, vehicle_transmission_time[vehicle_index, -1]))

                    if transmission_rate != 0:
                        occupied_time = int(np.floor(data_size / transmission_rate))
                        if self._occuiped and occupied_time > 0:
                            start_time = int(self._time_slots.now() + 1)
                            end_time = int(self._time_slots.now() + occupied_time + 1)
                            if end_time < self._config.time_slot_number:
                                for i in range(start_time, end_time):
                                    self._occupied_power[edge_index][i] += vehicle_edge_transmission_power[vehicle_index][edge_index]
                            else:
                                for i in range(start_time, int(self._config.time_slot_number)):
                                    self._occupied_power[edge_index][i] += vehicle_edge_transmission_power[vehicle_index][edge_index]        

        reward_part_5_time = time.time() - time_start
        
        time_start = time.time()
        
        successful_serviced = np.zeros(self._config.edge_number + 1)        
        rewards = np.zeros(self._config.edge_number + 1)
        edge_task_requested_number = np.zeros(self._config.edge_number)
        average_transmision_time = 0
        # for value in vehicle_transmission_time[:, -1]:
        #     if value != punished_time and value != 0:
        #         average_transmision_time += value
        average_wired_transmission_time = 0        
        # for value in vehicle_wired_transmission_time[:, -1]:
        #     if value != punished_time and value != 0:
        #         average_wired_transmission_time += value
        average_execution_time = 0
        # for value in vehicle_execution_time[:, -1]:
        #     if value != punished_time and value != 0:
        #         average_execution_time += value
        
        for edge_index in range(self._config.edge_number):
            if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
                for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                    task_required_number += 1
                    task_service_time = vehicle_transmission_time[vehicle_index, -1] + vehicle_wired_transmission_time[vehicle_index, -1] + vehicle_execution_time[vehicle_index, -1]
                    if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                        average_transmision_time += vehicle_transmission_time[vehicle_index, -1]
                        average_wired_transmission_time += vehicle_wired_transmission_time[vehicle_index, -1]
                        average_execution_time += vehicle_execution_time[vehicle_index, -1]
                        successful_serviced[-1] += 1
                        successful_serviced_number += 1                
                    for e in range(self._config.edge_number):
                        if e != edge_index:
                            edge_task_requested_number[e] += 1
                            task_service_time = vehicle_transmission_time[vehicle_index, e] + vehicle_wired_transmission_time[vehicle_index, e] + vehicle_execution_time[vehicle_index, e]
                            if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                                successful_serviced[e] += 1

        rewards[-1] = successful_serviced[-1] / task_required_number
        for edge_index in range(self._config.edge_number):
            rewards[edge_index] = successful_serviced[edge_index] / task_required_number
            # rewards[edge_index] = successful_serviced[edge_index] / edge_task_requested_number[edge_index]
        for edge_index in range(self._config.edge_number):
            rewards[edge_index] = rewards[-1] - rewards[edge_index]
        
        # for edge_index in range(self._config.edge_number + 1):
        #     rewards[edge_index] = -(np.sum(vehicle_transmission_time[:, edge_index]) + np.sum(vehicle_wired_transmission_time[:, edge_index]) + np.sum(vehicle_execution_time[:, edge_index]))
        # for edge_index in range(self._config.edge_number):
        #     rewards[edge_index] = rewards[-1] - rewards[edge_index]
        
        # rewards[-1] = -(np.sum(vehicle_transmission_time[:, -1]) + np.sum(vehicle_wired_transmission_time[:, -1]) + np.sum(vehicle_execution_time[:, -1]))
        cumulative_reward = successful_serviced[-1] / task_required_number
        
        average_vehicle_SINR = np.sum(vehicle_SINR[:, -1])
        
        average_vehicle_intar_interference = np.sum(vehicle_intar_edge_inference[:, -1])
        average_vehicle_inter_interference = np.sum(vehicle_inter_edge_inference[:, -1])
        average_vehicle_interference = average_vehicle_intar_interference + average_vehicle_inter_interference
                
        # average_transmision_time = np.sum(vehicle_transmission_time[:, -1])
        # average_wired_transmission_time = np.sum(vehicle_wired_transmission_time[:, -1])
        # average_execution_time = np.sum(vehicle_execution_time[:, -1])
        average_service_time = average_transmision_time + average_wired_transmission_time + average_execution_time
        
        # print("vehicle_service_time: ", vehicle_service_time)
        # print("average_service_time: ", average_service_time)
        # print("task_required_number: ", task_required_number)
        # print("task_average_service_time: ", average_service_time / task_required_number)
        # print("rewards:", rewards)
        # print("vehicle_SINR: ", vehicle_SINR)
        
        # myapp.info(f"\ntime: {self._time_slots.now()}")
        # myapp.info(f"\ntask_required_number: {task_required_number}")
        # if self._occuiped:
        #     myapp.info(f"\noccupied_power: {self._occupied_power}")
        #     myapp.info(f"\noccupied_computing_resources: {self._occupied_computing_resources}")
        # myapp.info(f"\nvehicle_transmission_time:\n{vehicle_transmission_time[:, -1]}")
        # myapp.info(f"\nvehicle_wired_transmission_time:\n{vehicle_wired_transmission_time[:, -1]}")
        # myapp.info(f"\nvehicle_execution_time:\n{vehicle_execution_time[:, -1]}")

        reward_part_6_time = time.time() - time_start
        
        # print("reward_1: ", reward_part_1_time)
        # print("reward_2: ", reward_part_2_time)
        # print("reward_3: ", reward_part_3_time)
        # print("reward_4: ", reward_part_4_time)
        # print("reward_5: ", reward_part_5_time)
        # print("reward_6: ", reward_part_6_time)
        # print("reward_7: ", reward_part_7_time)
        # print("reward_8: ", reward_part_8_time)
        # print("reward_9: ", reward_part_9_time)
        
        # if rewards[-1] > -400:
        
        # myapp.debug(f"rewards: {rewards}")
        
        # myapp.debug(f"vehicle_transmission_time[:, -1]: {vehicle_transmission_time[:, -1]}")
        # myapp.debug(f"vehicle_wired_transmission_time[:, -1]: {vehicle_wired_transmission_time[:, -1]}")
        # myapp.debug(f"vehicle_execution_time[:, -1]: {vehicle_execution_time[:, -1]}")
        # myapp.debug(f"np.sum(vehicle_transmission_time[:, -1]) + np.sum(vehicle_wired_transmission_time[:, -1]) + np.sum(vehicle_execution_time[:, -1]): {np.sum(vehicle_transmission_time[:, -1]) + np.sum(vehicle_wired_transmission_time[:, -1]) + np.sum(vehicle_execution_time[:, -1])}")
        # myapp.debug(f"rewards[-1]: {rewards[-1]}")
        
        # myapp.debug(f"vehicle_service_time: {vehicle_service_time}")
        # myapp.debug(f"average_service_time: {average_service_time}")
        # myapp.debug(f"task_required_number: {task_required_number}")
        
        
        return rewards, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_required_number
        

    """Define the action spaces of edge in critic network."""
    def critic_network_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        critic_network_action_shape = (self._config.critic_network_action_size, )
        return specs.BoundedArray(
            shape=(critic_network_action_shape),
            dtype=float,
            minimum=np.zeros(critic_network_action_shape),
            maximum=np.ones(critic_network_action_shape),
            name='critic_actions',
        )

    """Define the gloabl observation spaces."""
    def observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
            observation_shape = (self._config.edge_number, observation_size)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * observation_size, ) 
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
            observation_shape = (self._config.edge_number, observation_size)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * observation_size, )
        return specs.BoundedArray(
            shape=observation_shape,
            dtype=float,
            minimum=np.zeros(observation_shape),
            maximum=np.ones(observation_shape),
            name='observations'
        )
    
    def edge_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
        observation_shape = (observation_size, )
        return specs.BoundedArray(
            shape=observation_shape,
            dtype=float,
            minimum=np.zeros(observation_shape),
            maximum=np.ones(observation_shape),
            name='edge_observations'
        )
    
    """Define the gloabl action spaces."""
    def action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        action_shape = (self._config.edge_number, self._config.action_size)
        if self._flatten_space:
            action_shape = (self._config.edge_number * self._config.action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )

    def edge_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        action_shape = (self._config.action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )
    
    def reward_spec(self):
        """Define and return the reward space."""
        reward_shape = (self._config.reward_size, )
        return specs.Array(
            shape=reward_shape, 
            dtype=float, 
            name='rewards'
        )
    
    def _observation(self) -> np.ndarray:
        """Return the observation of the environment."""
        """
        observation_shape = (self._config.edge_number, self._config.observation_size)
        The observation space is the location, task size, and computation cycles of each vehicle, then the aviliable transmission power, and computation resoucers
        """
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
        # print("self._config.observation_size: ", self._config.observation_size)
        # print("self._for_mad5pg: ", self._for_mad5pg)
        # print("observation_size: ", observation_size)
        observation = np.zeros(shape=(self._config.edge_number, observation_size))
        
        for j in range(self._config.edge_number):
            vehicle_observed_index_within_edges = self._vehicle_observed_index_within_edges[j][self._time_slots.now()]
            vehicle_index_within_edges = self._vehicle_index_within_edges[j][self._time_slots.now()]
            vehicle_number_in_edge = len(vehicle_observed_index_within_edges)
            index = 0
            # if vehicle_number_in_edge != 3:
            #     print("vehicle_number_in_edge: ", len(vehicle_index_within_edges))
            for i in range(vehicle_number_in_edge):
                try:
                    vehicle_index = vehicle_observed_index_within_edges[i]
                    distance = self._distance_matrix[vehicle_index][j][self._time_slots.now()]
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index=vehicle_index).get_requested_task_by_slot_index(slot_index=self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index=task_index).get_data_size()
                    computing_cycles = self._task_list.get_task_by_index(task_index=task_index).get_computation_cycles()
                    delay_threshold = self._task_list.get_task_by_index(task_index=task_index).get_delay_threshold()
                    observation[j][index] = float(vehicle_index / self._config.vehicle_number)
                    index += 1
                    observation[j][index] = float(distance / self._config.communication_range)
                    index += 1
                    if task_index == -1:
                        observation[j][index] = 0
                        index += 1
                        observation[j][index] = 0
                        index += 1
                        observation[j][index] = 0
                        index += 1
                        observation[j][index] = 0
                        index += 1
                    else:
                        observation[j][index] = 1
                        index += 1
                        observation[j][index] = float((data_size - self._config.task_minimum_data_size) / (self._config.task_maximum_data_size - self._config.task_minimum_data_size))
                        index += 1
                        observation[j][index] = float((computing_cycles  - self._config.task_minimum_computation_cycles) / (self._config.task_maximum_computation_cycles - self._config.task_minimum_computation_cycles))
                        index += 1
                        observation[j][index] = float((delay_threshold - self._config.task_minimum_delay_thresholds) / (self._config.task_maximum_delay_thresholds - self._config.task_minimum_delay_thresholds))
                        index += 1
                except IndexError:
                    pass
            # print("index 1: ", index)
            index = self._config.vehicle_number_within_edges * 5
            for i in range(self._config.edge_number):
                edge_compuation_speed = self._edge_list.get_edge_by_index(edge_index=i).get_computing_speed()
                observation[j][index] = float((edge_compuation_speed - self._config.edge_minimum_computing_cycles)/ (self._config.edge_maximum_computing_cycles - self._config.edge_minimum_computing_cycles))
                index += 1
            # print("index 2: ", index)
            if self._occuiped and not self._for_mad5pg:
                for i in range(self._config.edge_number):
                    observation[j][index] = self._occupied_power[i][self._time_slots.now()] / ( 1.01 * self._config.edge_power)
                    index += 1
                    observation[j][index] = self._occupied_computing_resources[i][self._time_slots.now()] / ( 1.01 * self._config.edge_maximum_computing_cycles)
                    index += 1
            if self._for_mad5pg and not self._occuiped:
                observation[j][-2] = self._time_slots.now() / (self._config.time_slot_number - 1)
                observation[j][-1] = j / (self._config.edge_number - 1)
            # print("index 3: ", index)
            if self._occuiped and self._for_mad5pg:
                for i in range(self._config.edge_number):
                    observation[j][index] = self._occupied_power[i][self._time_slots.now()] / ( 1.01 * self._config.edge_power)
                    index += 1
                    observation[j][index] = self._occupied_computing_resources[i][self._time_slots.now()] / ( 1.01 * self._config.edge_maximum_computing_cycles)
                    index += 1
                observation[j][-2] = self._time_slots.now() / (self._config.time_slot_number - 1)
                observation[j][-1] = j / (self._config.edge_number - 1)
        if self._flatten_space:
            if self._occuiped:
                observation_size = self._config.observation_size
                if not self._for_mad5pg:
                    observation_size -= 2
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
                else:
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
            else:
                observation_size = self._config.observation_size - 2 * self._config.edge_number
                if not self._for_mad5pg:
                    observation_size -= 2
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
                else:
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
        
            
        return observation


Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""
    observations: NestedSpec
    edge_observations: NestedSpec
    critic_actions: NestedSpec
    actions: NestedSpec
    edge_actions: NestedSpec
    rewards: NestedSpec
    discounts: NestedSpec


def make_environment_spec(environment: vehicularNetworkEnv) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        edge_observations=environment.edge_observation_spec(),
        critic_actions=environment.critic_network_action_spec(),
        actions=environment.action_spec(),
        edge_actions=environment.edge_action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec())
    

def define_size_of_spaces(
    vehicle_number_within_edges: int,
    edge_number: int,
    task_assigned_number: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """The action space is transmison power, task assignment, and computing resources allocation"""
    action_size = vehicle_number_within_edges + vehicle_number_within_edges * edge_number
    
    """The observation space is the location, task size, computing cycles of each vehicle, then the aviliable transmission power, and computation resoucers"""
    observation_size = vehicle_number_within_edges * 6 + edge_number * 3 + 2
    
    """The reward space is the reward of each edge node and the gloabl reward
    reward[-1] is the global reward.
    reward[0:edge_number] are the edge rewards.
    """
    reward_size = edge_number + 1
    
    """Defined the shape of the action space in critic network"""
    critici_network_action_size = edge_number * action_size
    
    return action_size, observation_size, reward_size, critici_network_action_size

    
def init_distance_matrix_and_radio_coverage_matrix(
    env_config: env_config,
    vehicle_list: vehicleList,
    edge_list: edgeList,
) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]]]:
    """Initialize the distance matrix and radio coverage."""
    matrix_shpae = (env_config.vehicle_number, env_config.edge_number, env_config.time_slot_number)
    distance_matrix = np.zeros(matrix_shpae)
    channel_condition_matrix = [[[[] for _ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)] for _ in range(env_config.vehicle_number)]
    """Get the radio coverage information of each edge node."""
    vehicle_index_within_edges = [[[] for __ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)]
    vehicle_observed_index_within_edges = [[[] for __ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)]
    for i in range(env_config.vehicle_number):
        for j in range(env_config.edge_number):
            for k in range(env_config.time_slot_number):
                distance = vehicle_list.get_vehicle_by_index(i).get_distance_between_edge(k, edge_list.get_edge_by_index(j).get_edge_location())
                distance_matrix[i][j][k] = distance
                channel_condition_matrix[i][j][k] = compute_channel_gain(
                    rayleigh_distributed_small_scale_fading=generate_complex_normal_distribution(),
                    distance=distance,
                    path_loss_exponent=env_config.path_loss_exponent,
                )
                if distance_matrix[i][j][k] <= env_config.communication_range:
                    requested_task_index = vehicle_list.get_vehicle_by_index(i).get_requested_task_by_slot_index(k)
                    vehicle_observed_index_within_edges[j][k].append(i)
                    if requested_task_index != -1:
                        vehicle_index_within_edges[j][k].append(i)
    return distance_matrix, channel_condition_matrix, vehicle_index_within_edges, vehicle_observed_index_within_edges
    