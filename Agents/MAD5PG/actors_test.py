import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from absl.testing import absltest
from Agents.MAD5PG import actors
from Environment.environment import make_environment_spec
from Agents.MAD5PG.networks import make_policy_network
from Experiment.make_environment import get_default_environment
from Utilities.FileOperator import load_obj

class ActorTest(absltest.TestCase):


    def test_feedforward(self):

        environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/2022-08-02-15-20-31/init_environment_9f109dd07f5c45c48802b6e3d80d274b.pkl"
    
        environment = load_obj(environment_file_name)

        env_spec = make_environment_spec(environment)

        policy_networks = make_policy_network(env_spec.edge_actions)

        actor = actors.FeedForwardActor(
            policy_networks=policy_networks,
            
            edge_number=9,
            edge_action_size=6,
        )
        
        loop = EnvironmentLoop(environment, actor)
        loop.run(20)


if __name__ == '__main__':
    absltest.main()