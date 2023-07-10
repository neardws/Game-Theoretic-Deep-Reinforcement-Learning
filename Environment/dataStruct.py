
from operator import index
import re
import numpy as np
import pandas as pd
from typing import List


class timeSlots(object):
    """The set of discrete time slots of the system"""
    def __init__(
        self, 
        start: int, 
        end: int, 
        slot_length: int) -> None:
        """method to initialize the time slots
        Args:
            start: the start time of the system
            end: the end time of the system
            slot_length: the length of each time slot"""   
        self._start = start
        self._end = end
        self._slot_length = slot_length
        self._number = int((end - start + 1) / slot_length)
        self._now = start
        self.reset()

    def __str__(self) -> str:
        return f"now time: {self._now}, [{self._start} , {self._end}] with {self._slot_length} = {self._number} slots"
    
    def add_time(self) -> None:
        """method to add time to the system"""
        self._now += 1

    def is_end(self) -> bool:
        """method to check if the system is at the end of the time slots"""
        return self._now >= self._end

    def get_slot_length(self) -> int:
        """method to get the length of each time slot"""
        return int(self._slot_length)

    def get_number(self) -> int:
        return int(self._number)

    def now(self) -> int:
        return int(self._now)
    
    def get_start(self) -> int:
        return int(self._start)

    def get_end(self) -> int:
        return int(self._end)

    def reset(self) -> None:
        self._now = self._start
        
class task(object):
    def __init__(self, task_index: int, data_size: float, computation_cycles: float, delay_threshold: float) -> None:
        self._task_index = task_index
        self._data_size = data_size
        self._computation_cycles = computation_cycles
        self._delay_threshold = delay_threshold
    def get_task_index(self) -> int:
        return int(self._task_index)
    def get_data_size(self) -> float:
        return float(self._data_size)
    def get_computation_cycles(self) -> float:
        return float(self._computation_cycles)
    def get_delay_threshold(self) -> float:
        return float(self._delay_threshold)
    
class taskList(object):
    def __init__(
        self, 
        tasks_number: int, 
        minimum_data_size: float, 
        maximum_data_size: float,
        minimum_computation_cycles: float,
        maximum_computation_cycles: float,
        minimum_delay_thresholds: float,
        maximum_delay_thresholds: float,
        seed: int
    ) -> None:
        self._tasks_number = tasks_number
        self._minimum_data_size = minimum_data_size
        self._maximum_data_size = maximum_data_size
        self._minimum_computation_cycles = minimum_computation_cycles
        self._maximum_computation_cycles = maximum_computation_cycles
        self._minimum_delay_thresholds = minimum_delay_thresholds
        self._maximum_delay_thresholds = maximum_delay_thresholds
        self._seed = seed
        np.random.seed(seed)
        self._data_sizes = np.random.uniform(self._minimum_data_size , self._maximum_data_size, self._tasks_number)
        np.random.seed(seed)
        self._computation_cycles = np.random.uniform(self._minimum_computation_cycles, self._maximum_computation_cycles, self._tasks_number)
        np.random.seed(seed)
        self._delay_thresholds = np.random.uniform(self._minimum_delay_thresholds, self._maximum_delay_thresholds, self._tasks_number)
        self._task_list = [task(task_index, data_size, computation_cycle, delay_thresholod) for task_index, data_size, computation_cycle, delay_thresholod in zip(range(self._tasks_number), self._data_sizes, self._computation_cycles, self._delay_thresholds)]
    
    def get_task_list(self) -> List[task]:
        return self._task_list
    def get_task_by_index(self, task_index: int) -> task:
        return self._task_list[int(task_index)]
    
class location(object):
    """ the location of the node. """
    def __init__(self, x: float, y: float) -> None:
        """ initialize the location.
        Args:
            x: the x coordinate.
            y: the y coordinate.
        """
        self._x = x
        self._y = y

    def __str__(self) -> str:
        return f"x: {self._x}, y: {self._y}"
    def get_x(self) -> float:
        return self._x
    def get_y(self) -> float:
        return self._y
    def get_distance(self, location: "location") -> float:
        """ get the distance between two locations.
        Args:
            location: the location.
        Returns:
            the distance.
        """
        return np.math.sqrt(
            (self._x - location.get_x())**2 + 
            (self._y - location.get_y())**2
        )

class trajectory(object):
    """ the trajectory of the node. """
    def __init__(self, timeSlots: timeSlots, locations: List[location]) -> None:
        """ initialize the trajectory.
        Args:
            max_time_slots: the maximum number of time slots.
            locations: the location list.
        """
        self._locations = locations

        # if len(self._locations) != timeSlots.get_number():
        #     raise ValueError("The number of locations must be equal to the max_timestampes.")

    def __str__(self) -> str:
        return str([str(location) for location in self._locations])

    def get_location(self, nowTimeSlot: int) -> location:
        """ get the location of the timestamp.
        Args:
            timestamp: the timestamp.
        Returns:
            the location.
        """
        try:
            return self._locations[nowTimeSlot]
        except IndexError:
            return None

    def get_locations(self) -> List[location]:
        """ get the locations.
        Returns:
            the locations.
        """
        return self._locations

    def __str__(self) -> str:
        """ print the trajectory.
        Returns:
            the string of the trajectory.
        """
        print_result= ""
        for index, location in enumerate(self._locations):
            if index % 10 == 0:
                print_result += "\n"
            print_result += "(" + str(index) + ", "
            print_result += str(location.get_x()) + ", "
            print_result += str(location.get_y()) + ")"
        return print_result


class edge(object):
    """ the edge. """
    def __init__(
        self, 
        edge_index: int,
        power: float,
        bandwidth: float,
        computing_speed: float,
        communication_range: float,
        edge_x: float,
        edge_y: float
    ) -> None:
        self._edge_index = edge_index
        self._power = power
        self._bandwidth = bandwidth
        self._computing_speed = computing_speed
        self._communication_range = communication_range
        self._edge_location = location(edge_x, edge_y)
        
    def get_edge_index(self) -> int:
        return int(self._edge_index)
    def get_power(self) -> float:
        return float(self._power)
    def get_bandwidth(self) -> float:
        return float(self._bandwidth)
    def get_computing_speed(self) -> float:
        return float(self._computing_speed)
    def get_communication_range(self) -> float:
        return float(self._communication_range)
    def get_edge_location(self) -> location:
        return self._edge_location


class edgeList(object):
    def __init__(
        self,
        edge_number: int,
        power: float,
        bandwidth: float,
        minimum_computing_cycles: float,
        maximum_computing_cycles: float,
        communication_range: float,
        edge_xs: List[float],
        edge_ys: List[float],
        seed: int,
        uniformed: bool = True
        ) -> None:
        
        self._edge_number = edge_number
        self._power = power
        self._bandwidth = bandwidth
        self._minimum_computing_cycles = minimum_computing_cycles
        self._maximum_computing_cycles = maximum_computing_cycles
        self._communication_range = communication_range
        self._edge_xs = edge_xs
        self._edge_ys = edge_ys
        self._uniformed = uniformed
        self._seed = seed
        if uniformed:
            # np.random.seed(seed)
            # self._computing_speeds = np.random.uniform(self._minimum_computing_cycles, self._maximum_computing_cycles, self._edge_number)
            # 3 - 10
            self._computing_speeds = [3.0 * 1e9, 10.0 * 1e9, 3.0 * 1e9, 10.0 * 1e9, 6.0 * 1e9, 10.0 * 1e9, 3.0 * 1e9, 10.0 * 1e9, 3.0 * 1e9]
            # 1 - 10
            # self._computing_speeds = [1.0 * 1e9, 8.0 * 1e9, 1.0 * 1e9, 10.0 * 1e9, 4.0 * 1e9, 10.0 * 1e9, 1.0 * 1e9, 8.0 * 1e9, 1.0 * 1e9]
            # 2 - 10
            # self._computing_speeds = [2.0 * 1e9, 9.0 * 1e9, 2.0 * 1e9, 10.0 * 1e9, 5.0 * 1e9, 10.0 * 1e9, 2.0 * 1e9, 9.0 * 1e9, 2.0 * 1e9]
            # 4 - 10
            # self._computing_speeds = [4.5 * 1e9, 10.0 * 1e9, 4.5 * 1e9, 10.0 * 1e9, 7.0 * 1e9, 10.0 * 1e9, 4.5 * 1e9, 10.0 * 1e9, 4.5 * 1e9]
            # 5 - 10
            # self._computing_speeds = [6 * 1e9, 10.0 * 1e9, 6 * 1e9, 10.0 * 1e9, 8.0 * 1e9, 10.0 * 1e9, 6 * 1e9, 10.0 * 1e9, 6 * 1e9]
            self._edge_list = [edge(edge_index, self._power, self._bandwidth, computing_speed, self._communication_range, edge_x, edge_y) for edge_index, computing_speed, edge_x, edge_y in zip(range(edge_number), self._computing_speeds, self._edge_xs, self._edge_ys)]
        else:
            pass
        
    def get_edge_list(self) -> List[edge]:
        return self._edge_list
    def get_edge_by_index(self, edge_index: int) -> edge:
        return self._edge_list[int(edge_index)]


class vehicle(object):
    """" the vehicle. """
    def __init__(
        self, 
        vehicle_index: int,
        vehicle_trajectory: trajectory,
        slot_number: int,
        task_number: int,
        task_request_rate: float,
        seed: int,
    ) -> None:
        self._vehicle_index = vehicle_index
        self._vehicle_trajectory = vehicle_trajectory        
        self._slot_number = slot_number
        self._task_number = task_number
        self._task_request_rate = task_request_rate
        self._seed = seed
        
        self._requested_tasks = self.tasks_requested()

    def get_vehicle_index(self) -> int:
        return int(self._vehicle_index)
    def get_requested_tasks(self) -> List[int]:
        return self._requested_tasks
    def get_requested_task_by_slot_index(self, slot_index: int) -> int:
        return self._requested_tasks[slot_index]
    def get_vehicle_location(self, nowTimeSlot: int) -> location:
        return self._vehicle_trajectory.get_location(nowTimeSlot)
    def get_distance_between_edge(self, nowTimeSlot: int, edge_location: location) -> float:
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)
    def get_vehicle_trajectory(self) -> trajectory:
        return self._vehicle_trajectory

    def tasks_requested(self) -> List[int]:
        requested_task_number = int(self._slot_number * self._task_request_rate)
        requested_tasks = np.zeros(self._slot_number)
        for index in range(self._slot_number):
            requested_tasks[index] = -1
        if requested_task_number == self._slot_number:
            task_requested_time_slot_index = list(range(self._slot_number))
        else:
            np.random.seed(self._seed)
            task_requested_time_slot_index = list(np.random.choice(self._slot_number, requested_task_number, replace=False))
        np.random.seed(self._seed)
        task_requested_task_index = list(np.random.choice(self._task_number, requested_task_number, replace=True))
        for i in range(len(task_requested_time_slot_index)):
            requested_tasks[task_requested_time_slot_index[i]] = task_requested_task_index[i]
        return requested_tasks

class vehicleList(object):
    def __init__(
        self,
        edge_number: int,
        communication_range: float,
        vehicle_number: int,
        time_slots: timeSlots,
        trajectories_file_name: str,        
        slot_number: int,
        task_number: int,
        task_request_rate: float,
        seeds: List[int]
    ) -> None:
        self._edge_number = edge_number
        self._communication_range = communication_range
        self._vehicle_number = vehicle_number
        self._vehicle_number_in_edge = int(self._vehicle_number / self._edge_number)
        self._trajectories_file_name = trajectories_file_name
        self._slot_number = slot_number
        self._task_number = task_number
        self._task_request_rate = task_request_rate
        self._seeds = seeds
        
        self._vehicle_trajectories = self.read_vehicle_trajectories(time_slots)

        self._vehicle_list = [
            vehicle(
                vehicle_index=vehicle_index, 
                vehicle_trajectory=vehicle_trajectory, 
                slot_number=self._slot_number, 
                task_number=self._task_number, 
                task_request_rate=self._task_request_rate, 
                seed=seed) 
            for vehicle_index, vehicle_trajectory, seed in zip(
                range(self._vehicle_number), self._vehicle_trajectories, self._seeds)
        ]
    
    def get_vehicle_number(self) -> int:
        return int(self._vehicle_number)
    def get_slot_number(self) -> int:
        return int(self._slot_number)
    def get_task_number(self) -> int:
        return int(self._task_number)
    def get_task_request_rate(self) -> float:
        return float(self._task_request_rate)
    def get_vehicle_list(self) -> List[vehicle]:
        return self._vehicle_list
    def get_vehicle_by_index(self, vehicle_index: int) -> vehicle:
        return self._vehicle_list[int(vehicle_index)]
    
    def read_vehicle_trajectories(self, timeSlots: timeSlots) -> List[trajectory]:
        
        edge_number_in_width = int(np.sqrt(self._edge_number))
        vehicle_trajectories: List[trajectory] = []
        for i in range(edge_number_in_width):
            for j in range(edge_number_in_width):
                trajectories_file_name = self._trajectories_file_name + '_' + str(i) + '_' + str(j) + '.csv'
                df = pd.read_csv(
                    trajectories_file_name, 
                    names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)

                max_vehicle_id = df['vehicle_id'].max()
                
                selected_vehicle_id = []
                for vehicle_id in range(int(max_vehicle_id)):
                    new_df = df[df['vehicle_id'] == vehicle_id]
                    max_x = new_df['longitude'].max()
                    max_y = new_df['latitude'].max()
                    min_x = new_df['longitude'].min()
                    min_y = new_df['latitude'].min()
                    max_distance = np.sqrt((max_x - self._communication_range) ** 2 + (max_y - self._communication_range) ** 2)
                    min_distance = np.sqrt((min_x - self._communication_range) ** 2 + (min_y - self._communication_range) ** 2)
                    if max_distance < self._communication_range and min_distance < self._communication_range:
                        selected_vehicle_id.append(vehicle_id)
                
                if len(selected_vehicle_id) < self._vehicle_number_in_edge:
                    raise ValueError(f'i: {i}, j: {j}, len(selected_vehicle_id): {len(selected_vehicle_id)} Error: vehicle number in edge is less than expected')
                    
                for vehicle_id in selected_vehicle_id[ : self._vehicle_number_in_edge]:
                    new_df = df[df['vehicle_id'] == vehicle_id]
                    loc_list: List[location] = []
                    for row in new_df.itertuples():
                        # time = getattr(row, 'time')
                        x = getattr(row, 'longitude') + i * self._communication_range * 2
                        y = getattr(row, 'latitude') + j * self._communication_range * 2
                        loc = location(x, y)
                        loc_list.append(loc)
                    new_vehicle_trajectory: trajectory = trajectory(
                        timeSlots=timeSlots,
                        locations=loc_list
                    )
                    vehicle_trajectories.append(new_vehicle_trajectory)

        return vehicle_trajectories

