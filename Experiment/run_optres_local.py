import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Agents.Comparable_Algorithms.RA.actors import FeedForwardActor
from Utilities.FileOperator import load_obj

def main(_):

    # different scenario
    # scneario 1
    environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_1/local_environment_b6447224a61e446183f13dd40a04b17b.pkl"
    # scneario 2
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_2/local_environment_f1c365156c98462b9ae0b920d0063533.pkl"
    # scenario 3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_3/local_environment_0ff6ea4dcd184438aeb3389520f60aa9.pkl"
    # scenario 4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_4/local_environment_1bc5da3127734abc9d015bccf84bc1c0.pkl"

    # different compuation resources 
    # CPU 1-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/1GHz/local_environment_b80ebdb0027045288b59f66247950cb0.pkl"
    # CPU 2-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/2GHz/local_environment_62b140db2ca04cac84eab306ba2323d2.pkl"
    # CPU 4-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/4GHz/local_environment_aa0f303f501d427296ef9c93a2261868.pkl"
    # CPU 5-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/5GHz/local_environment_68eaeca4ef604e68b4753ad37530e431.pkl"
    
    # different task number
    # 0.3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_3/local_environment_590cd268b35a4e79b5b5216ee06e9ef3.pkl"
    # 0.4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_4/local_environment_00a13ba8916649c08c02f374dff640df.pkl"
    # 0.6
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_6/local_environment_05727ef5311540ca84b4c596a73987cd.pkl"
    # 0.7
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_7/local_environment_c2ea75aa7cce404e9d9f8af15d49369f.pkl"
    
    environment = load_obj(environment_file_name)

    actor = FeedForwardActor(
        agent_number=9,
        agent_action_size=27,
    )
    
    loop = EnvironmentLoop(environment, actor)
    loop.run(4100)

