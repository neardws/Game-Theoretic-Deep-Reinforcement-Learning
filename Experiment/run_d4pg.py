import sys

sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Agents.D4PG_new.agent import D4PG
from Agents.D4PG_new.networks import make_default_networks
from Environment.environment import make_environment_spec
from Utilities.FileOperator import load_obj

def main(_):
    
    # different scenario
    # scneario 1
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_1/global_environment_72604a5e5b364143b36131abaffb8b31.pkl"
    # scneario 2
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_2/global_environment_f37a4dc9bd164b529324908d12ca5c40.pkl"
    # scenario 3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_3/global_environment_96e3e39e202547ce9cfd06dc12604a71.pkl"
    # scenario 4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_4/global_environment_a17bdc9ba084473ab50abdc101037a0e.pkl"

    
    # different compuation resources 
    # CPU 1-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/1GHz/global_environment_5307f0921f624497a364cdfd1072f393.pkl"
    # CPU 2-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/2GHz/global_environment_5981cf8e689a476d812ce3357f549493.pkl"
    # CPU 4-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/4GHz/global_environment_64289740b65c432eaefe6755ec880c52.pkl"
    # CPU 5-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/5GHz/global_environment_53e106efdf0548c5a7e42e1cf0a467d7.pkl"
    
    # different task number
    # 0.3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_3/global_environment_c2bca604b01a4fe4b379f45794f4654c.pkl"
    # 0.4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_4/global_environment_3a7afa78fe8443e5a5cc208125749166.pkl"
    # 0.6
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_6/global_environment_6ef83d1602544f9b9fbd255d633a2803.pkl"
    # 0.7
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_7/global_environment_24e747fc18fb4c1da6eb6de2b2581bff.pkl"
    
    environment = load_obj(environment_file_name)

    spec = make_environment_spec(environment)
        
    networks = make_default_networks(spec.actions)
    
    agent = D4PG(
        environment_file=environment_file_name,
        environment_spec=spec,
        networks=networks,
        batch_size=256,
        prefetch_size=4,
        min_replay_size=1000,
        max_replay_size=1000000,
        samples_per_insert=8.0,
        n_step=1,
        sigma=0.3,
        discount=0.996,
        target_update_period=100,
    )
    
    loop = EnvironmentLoop(environment, agent)
    loop.run(4100)

