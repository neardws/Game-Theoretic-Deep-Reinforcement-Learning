import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from absl.testing import absltest
from Environment.environment import vehicularNetworkEnv, make_environment_spec
from Environment.environmentConfig import vehicularNetworkEnvConfig
from Environment.dataStruct import get_vehicle_number
from Agents.RANDOM import actors

class ActorTest(absltest.TestCase):
    def test_feedforward(self):

        config = vehicularNetworkEnvConfig()
        config.vehicle_number = int(get_vehicle_number(config.trajectories_file_name))
        config.vehicle_seeds += [i for i in range(config.vehicle_number)]

        env = vehicularNetworkEnv(config)

        env_spec = make_environment_spec(env)

        actor = actors.RandomActor(
            spec=env_spec,
        )
        loop = EnvironmentLoop(env, actor)
        loop.run(20)


if __name__ == '__main__':
    absltest.main()