from Environment.environment import make_environment_spec
from Agents.MAD4PG.networks import make_default_networks
from Agents.MAD4PG.agent import D4PGConfig, D4PG
from Experiment.make_environment import get_default_environment
from environment_loop import EnvironmentLoop

def main(_):
    
    time_slots, task_list, vehicle_list, edge_list, distance_matrix, channel_condition_matrix, \
        vehicle_index_within_edges, environment_config, environment = get_default_environment()
    
    spec = make_environment_spec(environment)

    # Create the networks.
    networks = make_default_networks(
        action_spec=spec.edge_actions,
    )

    agent_config = D4PGConfig()

    agent = D4PG(
        config=agent_config,
        environment = environment,
        environment_spec=spec,
        networks=networks
    )

    # Create the environment loop used for training.
    train_loop = EnvironmentLoop(environment, agent, label='train_loop')

    train_loop.run(num_episodes=5000)
        