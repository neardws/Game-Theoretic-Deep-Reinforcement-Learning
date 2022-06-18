import dataclasses
from typing import List

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
    task_minimum_data_size: float = 1 * 1024 * 1024 # 1MB
    task_maximum_data_size: float = 10 * 1024 * 1024 # 10MB
    task_minimum_computation_cycles: float = 1000
    task_maximim_computation_cycles: float = 10000
    task_seed: int = 0
    
    """"Edge related."""
    edge_number: int = 9
    edge_power: float = 1000.0 # mW
    edge_bandwidth: float = 3.0  # MHz
    edge_minimum_computing_cycles: float = 1000
    edge_maximum_computing_cycles: float = 10000
    communication_range: float = 500.0  # meters
    map_length: float = 3000.0  # meters
    map_width: float = 3000.0  # meters
    edge_seed: int = 0
    
    
    """Vehicle related."""
    vehicle_number: int = 200
    trajectories_file_name: str = 'CSV/trajectories_20161116_0800_0850.csv'
    task_request_rate: float = 0.3
    vehicle_seeds: List[int] = dataclasses.field(default_factory=list)

    """Vehicle Trajectories Processor related."""
    trajectories_file_name: str = 'CSV/gps_20161116'
    longitude_min: float = 104.04565967220308
    latitude_min: float = 30.654605745741608
    trajectories_time_start: str = '2016-11-16 08:00:00'
    trajectories_time_end: str = '2016-11-16 08:05:00'
    trajectories_out_file_name: str = 'CSV/trajectories_20161116_0800_0850.csv'

    """V2I Transmission related."""
    white_gaussian_noise: int = -90  # dBm
    mean_channel_fading_gain: float = 2.0 
    second_moment_channel_fading_gain: float = 0.4
    path_loss_exponent: int = 3
    wired_transmission_rate: float = 100.0 # Mbps
    wired_transmission_discount: float = 0.0006667
