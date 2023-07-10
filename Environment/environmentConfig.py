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
    task_minimum_data_size: float = 0.01 * 1024 * 1024 * 8 # 1 MB
    task_maximum_data_size: float = 5 * 1024 * 1024 * 8 # 5 MB
    task_minimum_computation_cycles: float = 500
    task_maximum_computation_cycles: float = 500 # CPU cycles for processing 1-bit of data
    task_minimum_delay_thresholds: float = 5 # seconds
    task_maximum_delay_thresholds: float = 10 # seconds
    task_seed: int = 0
    
    """"Edge related."""
    edge_number: int = 9
    edge_power: float = 1000.0 # mW
    edge_bandwidth: float = 20.0  # MHz
    edge_minimum_computing_cycles: float = 3.0 * 1e9 # 3 GHz
    edge_maximum_computing_cycles: float = 10.0 * 1e9 # 10 GHz
    communication_range: float = 500.0  # meters
    map_length: float = 3000.0  # meters
    map_width: float = 3000.0  # meters
    edge_seed: int = 0
    
    """Vehicle related."""
    vehicle_number: Optional[int] = 27
    trajectories_file_name: str = 'CSV/trajectories_20161116_1300_1305'
    task_request_rate: float = 1
    vehicle_seeds: List[int] = dataclasses.field(default_factory=list)

    """V2I Transmission related."""
    white_gaussian_noise: int = -90  # dBm
    mean_channel_fading_gain: float = 2.0 
    second_moment_channel_fading_gain: float = 0.4
    path_loss_exponent: int = 3
    wired_transmission_rate: float = 50.0 * 1024 * 1024 # 500 Mbps
    wired_transmission_discount: float = 0.0006667
    
    """Action related."""
    vehicle_number_within_edges: Optional[int] = None
    task_assigned_number: int = 2
    action_size: Optional[int] = None
    observation_size: Optional[int] = None
    reward_size: Optional[int] = None
    critic_network_action_size: Optional[int] = None
