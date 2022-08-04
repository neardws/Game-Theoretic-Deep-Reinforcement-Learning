import launchpad as lp
from Environment.environment import make_environment_spec
from Agents.MAD4PG.networks import make_default_networks
from Agents.MAD4PG.agent_distributed import DistributedD4PG
from Utilities.FileOperator import load_obj


def main(_):
    
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/2022-08-02-15-20-31/init_environment_9f109dd07f5c45c48802b6e3d80d274b.pkl"
    
    environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/2022-08-03-20-01-43/init_environment_7936536bea754ae4a054c88902c9e45c.pkl"
    environment = load_obj(environment_file_name)
    
    spec = make_environment_spec(environment)    
    
    
    networks = make_default_networks(
        agent_number=9,
        action_spec=spec.edge_actions,
    )

    agent = DistributedD4PG(
        agent_number=9,
        agent_action_size=30,
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
        max_actor_steps=300*5000,
        log_every=5.0,
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        