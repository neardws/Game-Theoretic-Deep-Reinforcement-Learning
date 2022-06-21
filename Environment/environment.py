
"""Vehicular Network Environments."""
import time 
import dm_env
from dm_env import specs
from acme.types import NestedSpec
import numpy as np
from typing import List, Tuple, NamedTuple, Optional
import Environment.environmentConfig as env_config
from Environment.dataStruct import timeSlots, taskList, edgeList, vehicleList, edgeAction
from Environment.utilities import generate_channel_fading_gain, compute_channel_condition, compute_transmission_rate, compute_SINR, compute_SNR, compute_edge_reward_with_SNR

class vehicularNetworkEnv(dm_env.Environment):
    """Vehicular Network Environment built on the dm_env framework."""
    
    def __init__(
        self, 
        envConfig: Optional[env_config.vehicularNetworkEnvConfig] = None) -> None:
        """Initialize the environment."""
        if envConfig is None:
            self._config = env_config.vehicularNetworkEnvConfig()
        else:
            self._config = envConfig

        self._time_slots: timeSlots = timeSlots(
            start=self._config.time_slot_start,
            end=self._config.time_slot_end,
            slot_length=self._config.time_slot_length,
        )
        
        self._task_list: taskList = taskList(
            tasks_number=self._config.task_number,
            minimum_data_size=self._config.task_minimum_data_size,
            maximum_data_size=self._config.task_maximum_data_size,
            minimum_computation_cycles=self._config.task_minimum_computation_cycles,
            maximum_computation_cycles=self._config.task_maximum_computation_cycles,
            seed=self._config.task_seed,
        )
        
        self._vehicle_list: vehicleList = vehicleList(
            vehicle_number=self._config.vehicle_number,
            time_slots=self._time_slots,
            trajectories_file_name=self._config.trajectories_file_name,
            slot_number=self._config.time_slot_number,
            task_number=self._config.task_number,
            task_request_rate=self._config.task_request_rate,
            seeds=self._config.vehicle_seeds,
        )
        
        self._edge_list: edgeList = edgeList(
            edge_number=self._config.edge_number,
            power=self._config.edge_power,
            bandwidth=self._config.edge_bandwidth,
            minimum_computing_cycles=self._config.edge_minimum_computing_cycles,
            maximum_computing_cycles=self._config.edge_maximum_computing_cycles,
            communication_range=self._config.communication_range,
            map_length=self._config.map_length,
            map_width=self._config.map_width,
            seed=self._config.edge_seed,
        )
        
        self._distance_matrix, self._channel_condition_matrix, self._vehicle_number_within_edges, \
            self._vehicle_index_within_edges, self._maximum_vehicle_number_within_edges = self.init_distance_matrix_and_radio_coverage_matrix()
        
        self._action_size, self._observation_size, self._reward_size, \
            self._critic_network_action_size = self._define_size_of_spaces()
        
        self._reward: np.ndarray = np.zeros(self._reward_size)
        
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        self._vehicle_SINR = np.zeros((self._config.vehicle_number,))
        self._vehicle_transmission_time = np.zeros((self._config.vehicle_number,))
        self._vehicle_execution_time = np.zeros((self._config.vehicle_number,))
        self._vehicle_wired_transmission_time = np.zeros((self._config.vehicle_number,))
        
        self._vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number,))
        self._vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number,))
        
        self._vehicle_edge_channel_condition = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_task_assignment = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_computation_resources = np.zeros((self._config.vehicle_number, self._config.edge_number))
        
        self._reset_next_step: bool = True
        
    def _define_size_of_spaces(self) -> Tuple[int, int, int, int]:
        """The action space is transmison power, task assignment, and computing resources allocation"""
        action_size = self._maximum_vehicle_number_within_edges * 3
        
        """The observation space is the location, task size, computing cycles of each vehicle, then the aviliable transmission power, and computation resoucers"""
        observation_size = self._maximum_vehicle_number_within_edges * 3 + 2
        
        """The reward space is the reward of each edge node and the gloabl reward
        reward[-1] is the global reward.
        reward[0:edge_number] are the edge rewards.
        """
        reward_size = self._config.edge_number + 1
        
        """Defined the shape of the action space in critic network"""
        critici_network_action_size = self._config.edge_number * action_size
        
        return int(action_size), int(observation_size), int(reward_size), int(critici_network_action_size)
        
    def reset(self) -> dm_env.TimeStep:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        Returns the first `TimeStep` of a new episode.
        """
        self._time_slots.reset()
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        self._vehicle_SINR = np.zeros((self._config.vehicle_number,))
        self._vehicle_transmission_time = np.zeros((self._config.vehicle_number,))
        self._vehicle_execution_time = np.zeros((self._config.vehicle_number,))
        self._vehicle_wired_transmission_time = np.zeros((self._config.vehicle_number,))
        
        self._vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number,))
        self._vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number,))
        
        self._vehicle_edge_channel_condition = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_task_assignment = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_computation_resources = np.zeros((self._config.vehicle_number, self._config.edge_number))
        
        self._reset_next_step = False
        return dm_env.restart(observation=self._observation())

    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """        
        if self._reset_next_step:
            return self.reset()
        self.compute_reward(action)
        observation = self._observation()
        # check for termination
        if self._time_slots.is_end():
            self._reset_next_step = True
            return dm_env.termination(observation=observation, reward=self._reward) 
        self._time_slots.add_time()
        return dm_env.transition(observation=observation, reward=self._reward)

    def transform_action_array_to_actions(self, action: np.ndarray) -> List[edgeAction]:
        """Transform the action array to the actions of edge nodes.
        Args:
            action: the action of the agent.
                the shape of the action is (edge_number, action_size)
        Returns:
            actions: the list of edge node actions.
        """ 
        edge_actions: List[edgeAction] = [
            self.generate_edge_action_from_np_array(
                now_time=self._time_slots.now(),
                edge_index=i,
                maximum_vehicle_number=self._maximum_vehicle_number_within_edges,
                network_output=action[i, :],
                action_time=self._time_slots.now(),
            ) for i in range(self._config.edge_number)
        ]
        
        return edge_actions

    def compute_reward(
        self,
        action: np.ndarray,
    ):
        actions = np.array(action)
        
        self._vehicle_SINR = np.zeros((self._config.vehicle_number,))
        self._vehicle_transmission_time = np.zeros((self._config.vehicle_number,))
        self._vehicle_execution_time = np.zeros((self._config.vehicle_number,))
        self._vehicle_wired_transmission_time = np.zeros((self._config.vehicle_number,))
        
        self._vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number,))
        self._vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number,))
        
        self._vehicle_edge_channel_condition = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_task_assignment = np.zeros((self._config.vehicle_number, self._config.edge_number))
        self._vehicle_edge_computation_resources = np.zeros((self._config.vehicle_number, self._config.edge_number))    
        
        for edge_index in range(self._config.edge_number):
            
            tasks_number_within_edge = int(self._vehicle_number_within_edges[edge_index][self._time_slots.now()])
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            
            transmission_power_allocation = np.array(actions[edge_index, : int(self._maximum_vehicle_number_within_edges)])
            task_assignment = np.array(actions[edge_index, int(self._maximum_vehicle_number_within_edges) : int(self._maximum_vehicle_number_within_edges) * 2])
            
            input_array = transmission_power_allocation[: tasks_number_within_edge]
            power_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
            
            edge_power = the_edge.get_power()
            edge_occupied_power = self._occupied_power[edge_index][self._time_slots.now()]
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                transmission_power = power_allocation[i]
                self._vehicle_edge_transmission_power[vehicle_index][edge_index] = transmission_power * (edge_power - edge_occupied_power)
            
            edge_number = self._config.edge_number
            
            task_assignment = task_assignment[: tasks_number_within_edge]
                        
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                if task_assignment[i] == 0:
                    task_assignment[i] = 0.01
                if task_assignment[i] == 1:
                    task_assignment[i] = 0.09
                
                processing_edge_index = int(np.floor(task_assignment[i] / (1 / edge_number)))
                self._vehicle_edge_task_assignment[vehicle_index][processing_edge_index] = 1
            
                if processing_edge_index != edge_index:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                    self._vehicle_wired_transmission_time[vehicle_index] = data_size / self._config.wired_transmission_rate * self._config.wired_transmission_discount * \
                            the_edge.get_edge_location().get_distance(self._edge_list.get_edge_by_index(processing_edge_index).get_edge_location())
        
        for edge_index in range(self._config.edge_number):

            edge_computing_speed = self._edge_list.get_edge_by_index(edge_index).get_computing_speed()
            edge_occupied_computing_speed = self._occupied_computing_resources[edge_index][self._time_slots.now()]
            computation_resource_allocation = np.array(actions[edge_index, int(self._maximum_vehicle_number_within_edges) * 2: ] )
            
            task_assignment_number = self._vehicle_edge_task_assignment[:, edge_index].sum()
            input_array = computation_resource_allocation[: int(task_assignment_number)]
            computation_resource_allocation = np.exp(input_array) / np.sum(np.exp(input_array))
            
            for vehicle_index, computation_resource in zip(np.where(self._vehicle_edge_task_assignment[:, edge_index] == 1)[0], computation_resource_allocation):
                self._vehicle_edge_computation_resources[vehicle_index][edge_index] = computation_resource * (edge_computing_speed - edge_occupied_computing_speed)
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                if self._vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                    self._vehicle_execution_time[vehicle_index] = float(data_size * computation_cycles / self._vehicle_edge_computation_resources[vehicle_index][edge_index])
                else:
                    self._vehicle_execution_time[vehicle_index] = self._config.time_slot_number
                if self._vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                    occupied_time = np.floor(data_size * computation_cycles / self._vehicle_edge_computation_resources[vehicle_index][edge_index])
                    if occupied_time > 0:
                        if self._time_slots.now() + int(occupied_time) < self._config.time_slot_number:
                            for i in range(int(occupied_time)):
                                self._occupied_computing_resources[edge_index][self._time_slots.now() + i] += self._vehicle_edge_computation_resources[vehicle_index][edge_index]
                        else:
                            for i in range(int(self._config.time_slot_number - 1 - self._time_slots.now())):
                                self._occupied_computing_resources[edge_index][self._time_slots.now() + i] += self._vehicle_edge_computation_resources[vehicle_index][edge_index]
        
        """Compute the inference"""
        for edge_index in range(self._config.edge_number):
            
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            for vehicle_index in vehicle_index_within_edge:
                channel_condition = self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()]
                for other_vehicle_index in vehicle_index_within_edge:
                    if other_vehicle_index != vehicle_index:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        if other_channel_condition < channel_condition:
                            self._vehicle_intar_edge_inference[vehicle_index] += other_channel_condition * self._vehicle_edge_transmission_power[other_vehicle_index][edge_index]
            
            
            for other_edge_index in range(self._config.edge_number):
                if other_edge_index != edge_index:
                    vehicle_index_within_other_edge = self._vehicle_index_within_edges[other_edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_other_edge:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        self._vehicle_inter_edge_inference[vehicle_index] += other_channel_condition * self._vehicle_edge_transmission_power[other_vehicle_index][other_edge_index]

        """Compute the SINR and transimission time"""
        for edge_index in range(self._config.edge_number):
            for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                self._vehicle_SINR[vehicle_index] = compute_SINR(
                    white_gaussian_noise=self._config.white_gaussian_noise, 
                    channel_condition=self._vehicle_edge_channel_condition[vehicle_index][edge_index],
                    transmission_power=self._vehicle_edge_transmission_power[vehicle_index][edge_index],
                    intra_edge_interference=self._vehicle_intar_edge_inference[vehicle_index],
                    inter_edge_interference=self._vehicle_inter_edge_inference[vehicle_index],)
                transmission_rate = compute_transmission_rate(
                    SINR=self._vehicle_SINR[vehicle_index], 
                    bandwidth=self._config.edge_bandwidth)
                if transmission_rate != 0:
                    self._vehicle_transmission_time[vehicle_index] = float(data_size / transmission_rate)
                else:
                    self._vehicle_transmission_time[vehicle_index] = self._config.time_slot_number
                if transmission_rate != 0:
                    occupied_time = np.floor(data_size / transmission_rate)
                    if occupied_time > 0:
                        if self._time_slots.now() + int(occupied_time) < self._config.time_slot_number:
                            for i in range(int(occupied_time)):
                                self._occupied_power[edge_index][self._time_slots.now() + i] += self._vehicle_edge_transmission_power[vehicle_index][edge_index]
                        else:
                            for i in range(int(self._config.time_slot_number - 1 - self._time_slots.now())):
                                self._occupied_power[edge_index][self._time_slots.now() + i] += self._vehicle_edge_transmission_power[vehicle_index][edge_index]        
        
        self._reward[-1] = - (np.sum(self._vehicle_transmission_time) + np.sum(self._vehicle_wired_transmission_time) + np.sum(self._vehicle_execution_time))
        
        for edge_index in range(self._config.edge_number):
            edge_reward = 0
            for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                SNR = compute_SNR(
                        white_gaussian_noise=self._config.white_gaussian_noise, 
                        channel_condition=self._vehicle_edge_channel_condition[vehicle_index][edge_index],\
                        transmission_power=self._vehicle_edge_transmission_power[vehicle_index][edge_index],
                        intra_edge_interference=self._vehicle_intar_edge_inference[vehicle_index],
                )
                first_reward = compute_edge_reward_with_SNR(
                    SNR=SNR, 
                    bandwidth=self._config.edge_bandwidth,
                    data_size=data_size)
                
                edge_reward += first_reward - self._vehicle_wired_transmission_time[vehicle_index] - self._vehicle_execution_time[vehicle_index]

            self._reward[edge_index] = edge_reward


    """Define the action spaces of edge in critic network."""
    def critic_network_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        critic_network_action_shape = (self._critic_network_action_size, )
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
        observation_shape = (int(self._config.edge_number), int(self._observation_size))
        return specs.BoundedArray(
            shape=observation_shape,
            dtype=float,
            minimum=np.zeros(observation_shape),
            maximum=np.ones(observation_shape),
            name='observations'
        )
    
    def edge_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        observation_shape = (int(self._observation_size), )
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
        action_shape = (self._config.edge_number, self._action_size)
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )

    def edge_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        action_shape = (self._action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )
    
    def reward_spec(self):
        """Define and return the reward space."""
        return specs.Array(
            shape=(self._reward_size,), 
            dtype=float, 
            name='rewards'
        )
    
    def _observation(self) -> np.ndarray:
        """Return the observation of the environment."""
        """
        observation_shape = (self._config.edge_number, self._observation_size)
        The observation space is the location, task size, and computation cycles of each vehicle, then the aviliable transmission power, and computation resoucers
        """
        observation = np.zeros((self._config.edge_number, self._observation_size,))
        for j in range(self._config.edge_number):
            vehicle_number_in_edge = self._vehicle_number_within_edges[j][self._time_slots.now()]
            index = 0
            for i in range(int(vehicle_number_in_edge)):
                vehicle_index = self._vehicle_index_within_edges[j][self._time_slots.now()][i]
                distance = self._distance_matrix[vehicle_index][j][self._time_slots.now()]
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index=vehicle_index).get_requested_task_by_slot_index(slot_index=self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index=task_index).get_data_size()
                computing_cycles = self._task_list.get_task_by_index(task_index=task_index).get_computation_cycles()
                observation[j][index] = float(distance / self._config.communication_range)
                index += 1
                observation[j][index] = float(data_size / self._config.task_maximum_data_size)
                index += 1
                observation[j][index] = float(computing_cycles / self._config.task_maximum_computation_cycles)
            observation[j][-2] = self._occupied_power[j][self._time_slots.now()]
            observation[j][-1] = self._occupied_computing_resources[j][self._time_slots.now()]
        return observation

    def init_distance_matrix_and_radio_coverage_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], int]:
        """Initialize the distance matrix and radio coverage."""
        matrix_shpae = (self._config.vehicle_number, self._config.edge_number, self._config.time_slot_number)
        distance_matrix = np.zeros(matrix_shpae)
        channel_condition_matrix = np.zeros(matrix_shpae)
        """Get the radio coverage information of each edge node."""
        vehicle_number_within_edges = np.zeros((self._config.edge_number, self._config.time_slot_number))
        vehicle_index_within_edges = [[[] for __ in range(self._config.time_slot_number)] for _ in range(self._config.edge_number)]

        for i in range(self._config.vehicle_number):
            for j in range(self._config.edge_number):
                for k in range(self._config.time_slot_number):
                    distance = self._vehicle_list.get_vehicle_by_index(i).get_distance_between_edge(k, self._edge_list.get_edge_by_index(j).get_edge_location())
                    distance_matrix[i][j][k] = distance
                    channel_condition_matrix[i][j][k] = compute_channel_condition(
                        generate_channel_fading_gain(self._config.mean_channel_fading_gain, self._config.second_moment_channel_fading_gain),
                        distance,
                        self._config.path_loss_exponent,
                    )
                    if distance_matrix[i][j][k] <= self._config.communication_range:
                        requested_task_index = self._vehicle_list.get_vehicle_by_index(i).get_requested_task_by_slot_index(k)
                        if requested_task_index != -1:
                            vehicle_number_within_edges[j][k] += 1
                            vehicle_index_within_edges[j][k].append(i)
        return distance_matrix, channel_condition_matrix, vehicle_number_within_edges, vehicle_index_within_edges, np.max(vehicle_number_within_edges)


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