import launchpad as lp
from Environment.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAD5PG.networks import make_default_MAD3PGNetworks
from Agents.MAD5PG.agent import MAD3PGConfig, MultiAgentDistributedDDPG
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
    
    print("environment._for_mad5pg: ", environment._for_mad5pg)
    spec = make_environment_spec(environment)

    agent_config = MAD3PGConfig(
        sigma=0.3,
    )

    # Create the networks.
    networks = make_default_MAD3PGNetworks(
        action_spec=spec.edge_actions,
        sigma=agent_config.sigma,
    )

    agent = MultiAgentDistributedDDPG(
        config=agent_config,
        environment_file_name=environment_file_name,
        environment_spec=spec,
        max_actor_steps=1500000,
        networks=networks,
        num_actors=10,
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        