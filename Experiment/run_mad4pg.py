import launchpad as lp
from Environment.environment_old import make_environment_spec
from Agents.MAD4PG.networks import make_default_networks
from Agents.MAD4PG.agent_distributed import DistributedD4PG
from Utilities.FileOperator import load_obj


def main(_):
    
    # different scenario
    # scneario 1
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_1/convex_environment_b6447224a61e446183f13dd40a04b17b.pkl"
    # scneario 2
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_2/convex_environment_f1c365156c98462b9ae0b920d0063533.pkl"
    # scenario 3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_3/convex_environment_0ff6ea4dcd184438aeb3389520f60aa9.pkl"
    # scenario 4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_4/convex_environment_1bc5da3127734abc9d015bccf84bc1c0.pkl"
    
    # different bandwidth 
    # bandwidth 10 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth10/convex_environment_0c3404cb7b094635b93478b7ed8414d4.pkl"
    # bandwidth 15 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth15/convex_environment_2aa46263ed1543a9b1724f2ae1e15517.pkl"
    # bandwidth 25 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth25/convex_environment_73e7ad98699f41ac9e940690c9bbf274.pkl"
    # bandwidth 30 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth30/convex_environment_99c0184f2a2f44ffa51e8570a3c56e44.pkl"
    
    # different power 
    # power 100 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/100mW/convex_environment_a7be501bbb0449e78ba3d18a915190f0.pkl"
    # power 550 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/550mW/convex_environment_d242a140f3a54b03af77a22a3e4698fe.pkl"
    # power 1450 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/1450mW/convex_environment_13da0cdb1f0f40849099c080b17e60bf.pkl"
    # power 1900 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/1900mW/convex_environment_4f77862d11a1479898416ea261c93b66.pkl"
    
    # different compuation resources 
    # CPU 1-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/1GHz/convex_environment_b80ebdb0027045288b59f66247950cb0.pkl"
    # CPU 2-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/2GHz/convex_environment_62b140db2ca04cac84eab306ba2323d2.pkl"
    # CPU 4-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/4GHz/convex_environment_aa0f303f501d427296ef9c93a2261868.pkl"
    # CPU 5-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/5GHz/convex_environment_68eaeca4ef604e68b4753ad37530e431.pkl"
    
    # different task number
    # 0.1
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_1/convex_environment_323579c648bf4169abcefc1c8036f79c.pkl"
    # 0.3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_3/convex_environment_590cd268b35a4e79b5b5216ee06e9ef3.pkl"
    # 0.4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_4/convex_environment_00a13ba8916649c08c02f374dff640df.pkl"
    # 0.6
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_6/convex_environment_05727ef5311540ca84b4c596a73987cd.pkl"
    # 0.7
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_7/convex_environment_c2ea75aa7cce404e9d9f8af15d49369f.pkl"
    # 0.9
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_9/convex_environment_02893eba453a40638713178e264ec23e.pkl"
    
    environment = load_obj(environment_file_name)
    
    spec = make_environment_spec(environment)    
    
    networks = make_default_networks(
        agent_number=9,
        action_spec=spec.edge_actions,
    )

    agent = DistributedD4PG(
        agent_number=9,
        agent_action_size=27,
        environment_file=environment_file_name,
        networks=networks,
        num_actors=10,
        environment_spec=spec,
        batch_size=256,
        prefetch_size=4,
        min_replay_size=1000,
        max_replay_size=1000000,
        samples_per_insert=8.0,
        n_step=1,
        sigma=0.3,
        discount=0.996,
        target_update_period=100,
        variable_update_period=1000,
        max_actor_steps=300*25000,
        log_every=5.0,
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        