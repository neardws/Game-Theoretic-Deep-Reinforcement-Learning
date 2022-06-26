
"""Vehicular Network Environments."""
import dm_env
from dm_env import specs
from acme.types import NestedSpec
import numpy as np
from typing import List, Tuple, NamedTuple, Optional
import Environment.environmentConfig as env_config
from Environment.dataStruct import timeSlots, taskList, edgeList, vehicleList
from Environment.utilities import generate_channel_fading_gain, compute_channel_condition, compute_transmission_rate, compute_SINR, compute_SNR, compute_edge_reward_with_SNR, cover_mW_to_W
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
        flatten_space: Optional[bool] = False,
        occuiped: Optional[bool] = False,
    ) -> None:
        """Initialize the environment."""
        if envConfig is None:
            from Environment.dataStruct import get_vehicle_number
            self._config = env_config.vehicularNetworkEnvConfig()
            self._config.vehicle_number = int(get_vehicle_number(self._config.trajectories_file_name) * self._config.vehicle_number_rate) 
            self._config.vehicle_seeds += [i for i in range(self._config.vehicle_number)]
            self._config.maximum_vehicle_number_within_edges = int(get_maximum_vehicle_number(
                env_config=self._config, 
                vehicle_list=vehicle_list, 
                edge_list=edge_list
            ))
            self._config.action_size, self._config.observation_size, self._config.reward_size, \
                self._config.critic_network_action_size = define_size_of_spaces(self._config.maximum_vehicle_number_within_edges, self._config.edge_number)
        else:
            self._config = envConfig
        
        if distance_matrix is None:
            self._distance_matrix, self._channel_condition_matrix, self._vehicle_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(
                env_config=self._config,
                vehicle_list=vehicle_list,
                edge_list=edge_list,
            )
        else:
            self._distance_matrix = distance_matrix
            self._channel_condition_matrix = channel_condition_matrix
            self._vehicle_index_within_edges = vehicle_index_within_edges
        
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
        
        self._reward, cumulative_reward, average_vehicle_interference, average_service_time, successful_serviced, task_required_number = self.compute_reward(action)
        
        observation = self._observation()
        # check for termination
        if self._time_slots.is_end():
            self._reset_next_step = True
            return dm_env.termination(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_interference, average_service_time, successful_serviced, task_required_number
        self._time_slots.add_time()
        return dm_env.transition(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_interference, average_service_time, successful_serviced, task_required_number

    def compute_reward(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        actions = np.array(action)
        
        # print("actions1: ", actions)
        if self._flatten_space:
            actions = np.reshape(actions, newshape=(self._config.edge_number, self._config.action_size))

        # print("actions2: ", actions)
        vehicle_SINR = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_execution_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_wired_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_service_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_interferences = np.zeros((self._config.vehicle_number, ))
        
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_task_assignment = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_computation_resources = np.zeros((self._config.vehicle_number, self._config.edge_number))    
        
        for edge_index in range(self._config.edge_number):
            try:
                vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            except IndexError:
                raise IndexError("edge_index: ", edge_index, "self._time_slots.now(): ", self._time_slots.now())
            tasks_number_within_edge = len(vehicle_index_within_edge)
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            
            transmission_power_allocation = np.array(actions[edge_index, : int(self._config.maximum_vehicle_number_within_edges)])
            task_assignment = np.array(actions[edge_index, int(self._config.maximum_vehicle_number_within_edges) : int(self._config.maximum_vehicle_number_within_edges) * 2])
            
            input_array = transmission_power_allocation[: tasks_number_within_edge]
            power_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
            
            edge_power = the_edge.get_power()
            edge_occupied_power = self._occupied_power[edge_index][self._time_slots.now()]
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                transmission_power = power_allocation[i]
                if self._occuiped:
                    if edge_power - edge_occupied_power <= 0:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = transmission_power * (edge_power - edge_occupied_power)
                else:
                    vehicle_edge_transmission_power[vehicle_index][edge_index] = transmission_power * edge_power
            
            edge_number = self._config.edge_number
            task_assignment = task_assignment[: tasks_number_within_edge]
                        
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                if task_assignment[i] < 0.01:
                    task_assignment[i] = 0.01
                if task_assignment[i] > 0.99:
                    task_assignment[i] = 0.99
                
                processing_edge_index = int(np.floor(task_assignment[i] / (1 / edge_number)))
                
                vehicle_edge_task_assignment[vehicle_index][processing_edge_index] = 1
            
                if processing_edge_index != edge_index:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                    wired_transmission_time = data_size / self._config.wired_transmission_rate * self._config.wired_transmission_discount * \
                            the_edge.get_edge_location().get_distance(self._edge_list.get_edge_by_index(processing_edge_index).get_edge_location())
                    for e in range(self._config.edge_number + 1):
                        vehicle_wired_transmission_time[vehicle_index, e] = wired_transmission_time
        
        for edge_index in range(self._config.edge_number):

            edge_computing_speed = self._edge_list.get_edge_by_index(edge_index).get_computing_speed()
            edge_occupied_computing_speed = self._occupied_computing_resources[edge_index][self._time_slots.now()]
            computation_resource_allocation = np.array(actions[edge_index, int(self._config.maximum_vehicle_number_within_edges) * 2: ] )
            
            task_assignment_number = vehicle_edge_task_assignment[:, edge_index].sum()
            input_array = computation_resource_allocation[: int(task_assignment_number)]
            computation_resource_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
            
            for vehicle_index, computation_resource in zip(np.where(vehicle_edge_task_assignment[:, edge_index] == 1)[0], computation_resource_allocation):
                if self._occuiped:
                    if edge_computing_speed - edge_occupied_computing_speed <= 0:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = computation_resource * (edge_computing_speed - edge_occupied_computing_speed)
                else:
                    vehicle_edge_computation_resources[vehicle_index][edge_index] = computation_resource * edge_computing_speed
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                if vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                    if float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index]) < self._config.time_slot_number:
                        vehicle_execution_time[vehicle_index, -1] = float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index])
                    else:
                        vehicle_execution_time[vehicle_index, -1] = self._config.time_slot_number
                else:
                    vehicle_execution_time[vehicle_index, -1] = self._config.time_slot_number
                    
                for e in range(self._config.edge_number):  # e is the edge node which do nothing
                    if e == edge_index:
                        vehicle_execution_time[vehicle_index, e] = self._config.time_slot_number
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
        
        """Compute the inference"""
        for edge_index in range(self._config.edge_number):
            
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            
            for vehicle_index in vehicle_index_within_edge:
                
                for other_edge_index in range(self._config.edge_number):
                    if other_edge_index != edge_index:
                        vehicle_index_within_other_edge = self._vehicle_index_within_edges[other_edge_index][self._time_slots.now()]
                        for other_vehicle_index in vehicle_index_within_other_edge:
                            other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                            inter_interference = other_channel_condition * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][other_edge_index])
                            vehicle_inter_edge_inference[vehicle_index, -1] += inter_interference
                            for e in range(self._config.edge_number):
                                if e == other_edge_index:
                                    vehicle_inter_edge_inference[vehicle_index, e] += 0
                                else:
                                    vehicle_inter_edge_inference[vehicle_index, e] += inter_interference    
                                
                channel_condition = self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()]
                for other_vehicle_index in vehicle_index_within_edge:
                    if other_vehicle_index != vehicle_index:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        if other_channel_condition < channel_condition:
                            vehicle_intar_edge_inference[vehicle_index, -1] += other_channel_condition * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][edge_index])
                for e in range(self._config.edge_number):
                    if e == edge_index:
                        vehicle_intar_edge_inference[vehicle_index, e] = 0
                    else:
                        vehicle_intar_edge_inference[vehicle_index, e] = vehicle_intar_edge_inference[vehicle_index, -1]
                
        """Compute the SINR and transimission time"""
        for edge_index in range(self._config.edge_number):
            for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                
                for e in range(self._config.edge_number):
                    if e == edge_index:
                        vehicle_transmission_time[vehicle_index, e] = self._config.time_slot_number
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
                            if float(data_size / transmission_rate) < self._config.time_slot_number:
                                vehicle_transmission_time[vehicle_index, e] = float(data_size / transmission_rate)
                            else:
                                vehicle_transmission_time[vehicle_index, e] = self._config.time_slot_number
                        else:
                            vehicle_transmission_time[vehicle_index, e] = self._config.time_slot_number
                
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
                    if float(data_size / transmission_rate) < self._config.time_slot_number:
                        vehicle_transmission_time[vehicle_index, -1] = float(data_size / transmission_rate)
                    else:
                        vehicle_transmission_time[vehicle_index, -1] = self._config.time_slot_number
                else:
                    vehicle_transmission_time[vehicle_index, -1] = self._config.time_slot_number
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

        task_requested_within_edge_number = 0
        successful_serviced = np.zeros(self._config.edge_number + 1)        
        rewards = np.zeros(self._config.edge_number + 1)
        
        for edge_index in range(self._config.edge_number):
            for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                task_requested_within_edge_number += 1
                task_service_time = vehicle_transmission_time[vehicle_index, -1] + vehicle_wired_transmission_time[vehicle_index, -1] + vehicle_execution_time[vehicle_index, -1]
                vehicle_service_time[vehicle_index] = task_service_time
                if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                    successful_serviced[-1] += 1
                
                vehicle_interferences[vehicle_index] = vehicle_intar_edge_inference[vehicle_index, -1] + vehicle_inter_edge_inference[vehicle_index, -1]
                
                for e in range(self._config.edge_number):
                    if e != edge_index:
                        task_service_time = vehicle_transmission_time[vehicle_index, e] + vehicle_wired_transmission_time[vehicle_index, e] + vehicle_execution_time[vehicle_index, e]
                        if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                            successful_serviced[e] += 1

        for edge_index in range(self._config.edge_number + 1):
            rewards[edge_index] = successful_serviced[edge_index] / task_requested_within_edge_number
        for edge_index in range(self._config.edge_number):
            rewards[edge_index] = rewards[-1] - rewards[edge_index]
        
        cumulative_reward = rewards[-1]
        average_vehicle_interference = np.sum(vehicle_interferences)
        average_service_time = np.sum(vehicle_service_time)
        successful_serviced = successful_serviced[-1]
        task_required_number =  task_requested_within_edge_number
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

        return rewards, cumulative_reward, average_vehicle_interference, average_service_time, successful_serviced, task_required_number
        

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
            observation_shape = (self._config.edge_number, self._config.observation_size)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * self._config.observation_size, )
        else:
            observation_shape = (self._config.edge_number, self._config.observation_size - 2)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * (self._config.observation_size - 2), )
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
            observation_shape = (self._config.observation_size, )
        else:
            observation_shape = (self._config.observation_size - 2, )
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
            observation = np.zeros(shape=(self._config.edge_number, self._config.observation_size))
        else:
            observation = np.zeros(shape=(self._config.edge_number, self._config.observation_size - 2))
        for j in range(self._config.edge_number):
            vehicle_index_within_edges = self._vehicle_index_within_edges[j][self._time_slots.now()]
            vehicle_number_in_edge = len(vehicle_index_within_edges)
            index = 0
            for i in range(vehicle_number_in_edge):
                vehicle_index = vehicle_index_within_edges[i]
                distance = self._distance_matrix[vehicle_index][j][self._time_slots.now()]
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index=vehicle_index).get_requested_task_by_slot_index(slot_index=self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index=task_index).get_data_size()
                computing_cycles = self._task_list.get_task_by_index(task_index=task_index).get_computation_cycles()
                observation[j][index] = float(distance / self._config.communication_range)
                index += 1
                observation[j][index] = float(data_size / self._config.task_maximum_data_size)
                index += 1
                observation[j][index] = float(computing_cycles / self._config.task_maximum_computation_cycles)
                index += 1
            if self._occuiped:
                observation[j][-2] = self._occupied_power[j][self._time_slots.now()] / ( 1.01 * self._config.edge_power)
                observation[j][-1] = self._occupied_computing_resources[j][self._time_slots.now()] / ( 1.01 * self._config.edge_maximum_computing_cycles)
        if self._flatten_space:
            if self._occuiped:
                observation = np.reshape(observation, newshape=(self._config.edge_number * self._config.observation_size, ))
            else:
                observation = np.reshape(observation, newshape=(self._config.edge_number * (self._config.observation_size - 2), ))
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
    maximum_vehicle_number_within_edges: int,
    edge_number: int,
) -> Tuple[int, int, int, int]:
    """The action space is transmison power, task assignment, and computing resources allocation"""
    action_size = maximum_vehicle_number_within_edges * 3
    
    """The observation space is the location, task size, computing cycles of each vehicle, then the aviliable transmission power, and computation resoucers"""
    observation_size = maximum_vehicle_number_within_edges * 3 + 2
    
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
    channel_condition_matrix = np.zeros(matrix_shpae)
    """Get the radio coverage information of each edge node."""
    vehicle_index_within_edges = [[[] for __ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)]

    for i in range(env_config.vehicle_number):
        for j in range(env_config.edge_number):
            for k in range(env_config.time_slot_number):
                distance = vehicle_list.get_vehicle_by_index(i).get_distance_between_edge(k, edge_list.get_edge_by_index(j).get_edge_location())
                distance_matrix[i][j][k] = distance
                channel_condition_matrix[i][j][k] = compute_channel_condition(
                    generate_channel_fading_gain(env_config.mean_channel_fading_gain, env_config.second_moment_channel_fading_gain),
                    distance,
                    env_config.path_loss_exponent,
                )
                if distance_matrix[i][j][k] <= env_config.communication_range:
                    requested_task_index = vehicle_list.get_vehicle_by_index(i).get_requested_task_by_slot_index(k)
                    if requested_task_index != -1:
                        vehicle_index_within_edges[j][k].append(i)
    return distance_matrix, channel_condition_matrix, vehicle_index_within_edges
    
    
def get_maximum_vehicle_number(
        env_config: env_config,
        vehicle_list: vehicleList,
        edge_list: edgeList,
    ) -> int:
        vehicle_number_within_edges = np.zeros((env_config.edge_number, env_config.time_slot_number))
        for k in range(env_config.time_slot_number):
            for j in range(env_config.edge_number):
                for i in range(env_config.vehicle_number):
                    distance = vehicle_list.get_vehicle_by_index(i).get_distance_between_edge(k, edge_list.get_edge_by_index(j).get_edge_location())
                    if distance <= env_config.communication_range:
                        requested_task_index = vehicle_list.get_vehicle_by_index(i).get_requested_task_by_slot_index(k)
                        if requested_task_index != -1:
                            vehicle_number_within_edges[j][k] += 1
        return np.max(vehicle_number_within_edges)