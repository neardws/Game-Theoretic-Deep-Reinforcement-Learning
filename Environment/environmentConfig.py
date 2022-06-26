import dataclasses
from typing import List, Optional

from Environment.dataStruct import task

@dataclasses.dataclass
class vehicularNetworkEnvConfig:
    """Configuration for the vehicular network environment."""
    
    """Time slot related."""
    time_slot_start: int = 0
    time_slot_end: int = 299
    time_slot_number: int = 300
    time_slot_length: int = 1
    
    """Task related."""
    task_number: int = 100
    task_minimum_data_size: float = 1 * 1024 * 1024 # 1 MB
    task_maximum_data_size: float = 50 * 1024 * 1024 # 50 MB
    task_minimum_computation_cycles: float = 10
    task_maximum_computation_cycles: float = 250 # CPU cycles for processing 1-Byte of data
    task_seed: int = 0
    
    """"Edge related."""
    edge_number: int = 9
    edge_power: float = 1000.0 # mW
    edge_bandwidth: float = 10.0  # MHz
    edge_minimum_computing_cycles: float = 1.0 * 1e9 # 1 GHz
    edge_maximum_computing_cycles: float = 4.0 * 1e9 # 4 GHz
    communication_range: float = 500.0  # meters
    map_length: float = 3000.0  # meters
    map_width: float = 3000.0  # meters
    edge_seed: int = 0
    
    """Vehicle related."""
    vehicle_number_rate: float = 0.3
    vehicle_number: Optional[int] = None 
    trajectories_file_name: str = 'CSV/trajectories_20161116_0800_0850.csv'
    task_request_rate: float = 0.2
    vehicle_seeds: List[int] = dataclasses.field(default_factory=list)

    """V2I Transmission related."""
    white_gaussian_noise: int = -90  # dBm
    mean_channel_fading_gain: float = 2.0 
    second_moment_channel_fading_gain: float = 0.4
    path_loss_exponent: int = 3
    wired_transmission_rate: float = 50.0 * 1024 * 1024 # 500 Mbps
    wired_transmission_discount: float = 0.0006667
    
    """Action related."""
    maximum_vehicle_number_within_edges: Optional[int] = None
    action_size: Optional[int] = None
    observation_size: Optional[int] = None
    reward_size: Optional[int] = None
    critic_network_action_size: Optional[int] = None
