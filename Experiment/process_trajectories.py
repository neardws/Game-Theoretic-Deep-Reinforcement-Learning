import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from Environment.utilities import vehicleTrajectoriesProcessor

if __name__ == "__main__":
    """Vehicle Trajectories Processor related."""
    trajectories_file_name: str = 'CSV/gps_20161116'
    longitude_min: float = 104.04565967220308
    latitude_min: float = 30.654605745741608
    trajectories_time_start: str = '2016-11-16 23:00:00'
    trajectories_time_end: str = '2016-11-16 23:05:00'
    trajectories_out_file_name: str = 'CSV/trajectories_20161116_2300_2305'
    edge_number: int = 9
    communication_range: float = 500
    
    processor = vehicleTrajectoriesProcessor(
        file_name=trajectories_file_name, 
        longitude_min=longitude_min, 
        latitude_min=latitude_min,
        edge_number=edge_number,
        map_width=3000.0,
        communication_range=communication_range,
        time_start=trajectories_time_start,
        time_end=trajectories_time_end, 
        out_file=trajectories_out_file_name,
    )    
    