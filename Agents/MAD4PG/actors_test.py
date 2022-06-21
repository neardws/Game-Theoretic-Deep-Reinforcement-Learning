import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from absl.testing import absltest
from Agents.MAD4PG import actors
from Environment.environment import vehicularNetworkEnv, make_environment_spec
from Environment.environmentConfig import vehicularNetworkEnvConfig
from Agents.MAD4PG.networks import make_policy_network
from Environment.dataStruct import get_vehicle_number

class ActorTest(absltest.TestCase):


    def test_feedforward(self):

        config = vehicularNetworkEnvConfig()
        config.vehicle_number = int(get_vehicle_number(config.trajectories_file_name))
        config.vehicle_seeds += [i for i in range(config.vehicle_number)]

        env = vehicularNetworkEnv(config)

        env_spec = make_environment_spec(env)

        policy_networks = [make_policy_network(env_spec.edge_actions) for _ in range(config.edge_number)]

        actor = actors.FeedForwardActor(
            policy_networks=policy_networks,
            
            edge_number=config.edge_number,
            edge_action_size=env._action_size,
        )
        loop = EnvironmentLoop(env, actor)
        loop.run(20)


if __name__ == '__main__':
    absltest.main()