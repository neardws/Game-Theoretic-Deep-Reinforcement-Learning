import sys

import scipy as sp
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Agents.DDPG.agent import DDPG, make_default_networks
from Environment.environment import make_environment_spec
from Utilities.FileOperator import load_obj

def main(_):
    
    # different scenario
    # scneario 1
    environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/2022-10-05-17-04-03/global_environment_72604a5e5b364143b36131abaffb8b31.pkl"
    # scneario 2
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_2/old_environment_b624df881ef4447ba4f64825347c4f62.pkl"
    # scenario 3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_3/old_environment_f64be56731574faf9d005c17db90874e.pkl"
    # scenario 4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_4/old_environment_e49cf126ed0749e6bf3835f6dbb27ff2.pkl"

    
    # different compuation resources 
    # CPU 1-10GHz
    # environment_file_name = ""
    # CPU 2-10GHz
    # environment_file_name = ""
    # CPU 4-10GHz
    # environment_file_name = ""
    # CPU 5-10GHz
    # environment_file_name = ""
    
    # different task number
    # 0.3
    # environment_file_name = ""
    # 0.4
    # environment_file_name = ""
    # 0.6
    # environment_file_name = ""
    # 0.7
    # environment_file_name = ""
    
    environment = load_obj(environment_file_name)

    spec = make_environment_spec(environment)
    print(spec.actions)
    print(spec.observations)
    
        
    observation_network, policy_network, critic_network = make_default_networks(spec.actions)
    
    agent = DDPG(
        environment_spec=spec,
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
    )
    
    loop = EnvironmentLoop(environment, agent)
    loop.run(5000)

