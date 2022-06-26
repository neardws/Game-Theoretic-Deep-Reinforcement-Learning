import launchpad as lp
from Environment.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAD4PG.networks import make_default_MAD3PGNetworks
from Agents.MAD4PG.agent import MAD3PGConfig, MultiAgentDistributedDDPG
from Experiment.make_environment import get_default_environment

def main(_):
    
    time_slots, task_list, vehicle_list, edge_list, distance_matrix, channel_condition_matrix, \
        vehicle_index_within_edges, environment_config, environment = get_default_environment()
    
    spec = make_environment_spec(environment)

    # Create the networks.
    networks = make_default_MAD3PGNetworks(
        action_spec=spec.edge_actions,
    )

    agent_config = MAD3PGConfig()

    agent = MultiAgentDistributedDDPG(
        config=agent_config,
        environment_factory=lambda x: vehicularNetworkEnv(
            envConfig = environment_config,
            time_slots = time_slots,
            task_list = task_list,
            vehicle_list = vehicle_list,
            edge_list = edge_list,
            distance_matrix = distance_matrix, 
            channel_condition_matrix = channel_condition_matrix, 
            vehicle_index_within_edges = vehicle_index_within_edges,    
        ),
        environment_spec=spec,
        max_actor_steps=1500000,
        networks=networks,
        num_actors=10,
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        