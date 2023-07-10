"""Generic actor implementation, using TensorFlow and Sonnet."""

from typing import Optional, List
from acme import adders
from acme import core
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from Log.logger import myapp
tfd = tfp.distributions


class FeedForwardActor(core.Actor):
    """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_one_networks: snt.Module,
        policy_two_networks: snt.Module,
        
        edge_number: int,
        edge_action_size: int,

        adder: Optional[adders.Adder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initializes the actor.

        Args:
            policy_network: A module which takes observations and outputs
                actions.
            adder: the adder object to which allows to add experiences to a
                dataset/replay buffer.
            variable_client: object which allows to copy weights from the learner copy
                of the policy to the actor copy (in case they are separate).
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._policy_one_networks = policy_one_networks
        self._policy_two_networks = policy_two_networks

        self._edge_number = edge_number
        self._edge_action_size = edge_action_size

    @tf.function(experimental_relax_shapes=True)
    def _policy(
        self, 
        observations: types.NestedTensor,
    ) -> types.NestedTensor:
        # # Add a dummy batch dimension and as a side effect convert numpy to TF.
        # Compute the policy, conditioned on the observation.
        # myapp.debug(f"observations: {np.array(observations)}")
        edge_one_actions = []
        edge_two_actions = []
        for i in range(self._edge_number):
            # myapp.debug(f"i: {i}")
            edge_observation = observations[i, :]
            # myapp.debug(f"edge_observation: {np.array(edge_observation)}")
            edge_batched_observation = tf2_utils.add_batch_dim(edge_observation)
            # myapp.debug(f"edge_batched_observation: {edge_batched_observation}")
            edge_one_policy = self._policy_one_networks(edge_batched_observation)
            edge_one_action = edge_one_policy.sample() if isinstance(edge_one_policy, tfd.Distribution) else edge_one_policy
            # myapp.debug(f"edge_action: {edge_action}")
            edge_one_actions.append(edge_one_action)
        
        input_edge_one_actions = tf.convert_to_tensor(edge_one_actions, dtype=tf.float64)
        input_edge_one_action = tf2_utils.add_batch_dim(tf.reshape(input_edge_one_actions, [self._edge_number * self._edge_one_action_size, ]))
        
        for i in range(self._edge_number):
            edge_observation = observations[i, :]
            edge_batched_observation = tf2_utils.add_batch_dim(edge_observation)
            edge_two_policy_input = tf2_utils.batch_concat(inputs=[edge_batched_observation, input_edge_one_action])
            edge_two_policy = self._policy_two_networks(edge_two_policy_input)
            edge_two_action = edge_two_policy.sample() if isinstance(edge_two_policy, tfd.Distribution) else edge_two_policy
            edge_two_actions.append(edge_two_action)
            
        edge_one_action_tensor = tf.convert_to_tensor(edge_one_actions, dtype=tf.float64)
        edge_two_action_tensor = tf.convert_to_tensor(edge_two_actions, dtype=tf.float64)
        # myapp.debug(f"edge_actions: {edge_actions}")
        action = tf.reshape(tf2_utils.batch_concat(inputs=[edge_one_action_tensor, edge_two_action_tensor]), [self._edge_number, self._edge_action_size])
        # myapp.debug(f"action: {action}")
        return action

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        # Pass the observation through the policy network.
        action = self._policy(
            observations=tf.convert_to_tensor(observation, dtype=tf.float64))
        # Return a numpy array with squeezed out batch dimension.
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)