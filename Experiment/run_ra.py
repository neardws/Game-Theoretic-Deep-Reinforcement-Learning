import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Environment.environment import make_environment_spec
from Agents.RANDOM.actors import RandomActor
from Utilities.FileOperator import load_obj

def main(_):

    # task_request_rate=0.3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/different_task_number/task_request_rate_0_3/init_environment_f80db9577b96498d89be3677d49d528e.pkl" 
    # task_request_rate=0.35
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/different_task_number/task_request_rate_0_3_5/init_environment_464cc239ae0b43b0a7ff61ac39a171c7.pkl" 
    # task_request_rate=0.4
    environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/different_task_number/task_request_rate_0_4/init_environment_6941aa5605e24de3a8e370cfa86dbb0d.pkl" 
    # task_request_rate=0.45
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/different_task_number/task_request_rate_0_4_5/init_environment_19d53b92cd9e4e1ea895d1a809848473.pkl" 
    # task_request_rate=0.5
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/different_task_number/task_request_rate_0_5/init_environment_524c2333e5474adfaf67b8d6c0fc7fd7.pkl" 
    
    
    environment = load_obj(environment_file_name)
    env_spec = make_environment_spec(environment)

    actor = RandomActor(
        spec=env_spec,
    )
    loop = EnvironmentLoop(environment, actor)
    loop.run(100)

