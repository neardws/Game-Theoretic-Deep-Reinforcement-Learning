import launchpad as lp
from Environment.dataStruct import get_vehicle_number
from Environment.environment import vehicularNetworkEnv, make_environment_spec
from Environment.environmentConfig import vehicularNetworkEnvConfig
from Agents.MAD4PG.networks import make_default_MAD3PGNetworks
from Agents.MAD4PG.agent import MAD3PGConfig, MultiAgentDistributedDDPG

def main(_):
    
    environment_config = vehicularNetworkEnvConfig()
    environment_config.vehicle_number = int(get_vehicle_number(environment_config.trajectories_file_name))
    environment_config.vehicle_seeds += [i for i in range(environment_config.vehicle_number)]

    environment = vehicularNetworkEnv(environment_config)

    spec = make_environment_spec(environment)

    # Create the networks.
    networks = make_default_MAD3PGNetworks(
        action_spec=spec.actions,
    )

    agent_config = MAD3PGConfig()

    agent = MultiAgentDistributedDDPG(
        config=agent_config,
        environment_factory=lambda x: vehicularNetworkEnv(environment_config),
        environment_spec=spec,
        max_actor_steps=5000,
        networks=networks,
        num_actors=10,
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        