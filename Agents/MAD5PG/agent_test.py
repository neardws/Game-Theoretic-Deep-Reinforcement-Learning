
"""Integration test for the distributed agent."""
import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
import acme
import launchpad as lp
from absl.testing import absltest
from Environment.environment import vehicularNetworkEnv, make_environment_spec
from Agents.MAD5PG.networks import make_default_MAD3PGNetworks
from Agents.MAD5PG.agent import MultiAgentDistributedDDPG, MAD3PGConfig
from Experiment.make_environment import get_default_environment


class DistributedAgentTest(absltest.TestCase):
    """Simple integration/smoke test for the distributed agent."""

    def test_control_suite(self):
        """Tests that the agent can run on the control suite without crashing."""

        time_slots, task_list, vehicle_list, edge_list, distance_matrix, channel_condition_matrix, \
        vehicle_index_within_edges, environment_config, environment = get_default_environment(for_mad5pg=True)
    
        spec = make_environment_spec(environment)

        # Create the networks.
        networks = make_default_MAD3PGNetworks(
            action_spec=spec.edge_actions,
        )
        
        agent_config = MAD3PGConfig(
            batch_size=32, 
            min_replay_size=32, 
            max_replay_size=1000,
        )

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
            networks=networks,
            num_actors=2,
        )
        
        program = agent.build()

        (learner_node,) = program.groups['learner']
        learner_node.disable_run()

        lp.launch(program, launch_type='test_mt', serialize_py_nodes=False)

        learner: acme.Learner = learner_node.create_handle().dereference()

        for _ in range(5):
            learner.step()


if __name__ == '__main__':
    absltest.main()