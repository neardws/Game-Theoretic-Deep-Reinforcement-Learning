import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from absl.testing import absltest
from Agents.MAD5PG import actors
from Environment.environment import make_environment_spec
from Agents.MAD5PG.networks import make_policy_network
from Experiment.make_environment import get_default_environment

class ActorTest(absltest.TestCase):


    def test_feedforward(self):

        time_slots, task_list, vehicle_list, edge_list, distance_matrix, channel_condition_matrix, \
        vehicle_index_within_edges, environment_config, environment = get_default_environment(for_mad5pg=True)

        env_spec = make_environment_spec(environment)

        policy_networks = make_policy_network(env_spec.edge_actions)

        actor = actors.FeedForwardActor(
            policy_networks=policy_networks,
            
            edge_number=environment_config.edge_number,
            edge_action_size=environment_config.action_size,
        )
        loop = EnvironmentLoop(environment, actor)
        loop.run(20)


if __name__ == '__main__':
    absltest.main()