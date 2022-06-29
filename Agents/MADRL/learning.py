"""MAD3PG learner implementation."""

import time
from typing import Dict, Iterator, List, Optional, Union, Sequence
import acme
from acme.tf import losses
from acme.tf import networks as acme_nets
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import tensorflow as tf
from acme import types
from Agents.MAD4PG.gradient import GradientTape
from Environment.dataStruct import edge
from Log.logger import myapp

Replicator = Union[snt.distribute.Replicator, snt.distribute.TpuReplicator]


class MAD3PGLearner(acme.Learner):
    """MAD3PG learner.

    This is the learning component of a D3PG agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        policy_networks: List[snt.Module],
        critic_networks: List[snt.Module],
        
        target_policy_networks: List[snt.Module],
        target_critic_networks: List[snt.Module],

        discount: float,
        target_update_period: int,
        dataset_iterator: Iterator[reverb.ReplaySample],
        
        observation_networks: List[types.TensorTransformation],
        target_observation_networks: List[types.TensorTransformation],

        policy_optimizers: List[snt.Optimizer],
        critic_optimizers: List[snt.Optimizer],
        
        clipping: bool = True,
        replicator: Optional[Replicator] = None,

        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        
        edge_number: Optional[int] = None,
        edge_action_size: Optional[int] = None,
    ):
        """Initializes the learner.

        Args:
        policy_network: the online (optimized) policy.
        critic_network: the online critic.
        target_policy_network: the target policy (which lags behind the online
            policy).
        target_critic_network: the target critic.
        discount: discount to use for TD updates.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        dataset_iterator: dataset to learn from, whether fixed or from a replay
            buffer (see `acme.datasets.reverb.make_reverb_dataset` documentation).
        observation_network: an optional online network to process observations
            before the policy and the critic.
        target_observation_network: the target observation network.
        policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
        critic_optimizer: the optimizer to be applied to the distributional
            Bellman loss.
        clipping: whether to clip gradients by global norm.
        replicator: Replicates variables and their update methods over multiple
        accelerators, such as the multiple chips in a TPU.
        counter: counter object used to keep track of steps.
        logger: logger object to be used by learner.
        checkpoint: boolean indicating whether to checkpoint the learner.
        """

        # Store online and target networks.
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks

        self._target_policy_networks = target_policy_networks
        self._target_critic_networks = target_critic_networks

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_networks = [tf2_utils.to_sonnet_module(observation_network) for observation_network in observation_networks]
        self._target_observation_networks = [tf2_utils.to_sonnet_module(target_observation_network) for target_observation_network in target_observation_networks]

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger('learner')

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Replicates Variables across multiple accelerators
        if not replicator:
            accelerator = get_first_available_accelerator_type()
            if accelerator == 'TPU':
                replicator = snt.distribute.TpuReplicator()
            else:
                replicator = snt.distribute.Replicator()

        self._replicator = replicator

        with replicator.scope():
            # Necessary to track when to update target networks.
            self._num_steps = tf.Variable(0, dtype=tf.int32)
            self._target_update_period = target_update_period

            # Create optimizers if they aren't given.
            self._policy_optimizers = policy_optimizers or [snt.optimizers.Adam(learning_rate=1e-5) for _ in range(edge_number)]
            self._critic_optimizers = critic_optimizers or [snt.optimizers.Adam(learning_rate=1e-4) for _ in range(edge_number)]

        # Batch dataset and create iterator.
        self._iterator = dataset_iterator

        # Expose the variables.
        self._variables = dict()
        
        for i in range(len(self._policy_networks)):
            policy_network_to_expose = snt.Sequential(
                [self._target_observation_networks[i], self._target_policy_networks[i]])
            self._variables[f'critic_network_{i}'] = self._target_critic_networks[i].variables
            self._variables[f'policy_network_{i}'] = policy_network_to_expose.variables
        
        # Create a checkpointer and snapshotter objects.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                subdirectory='mad3pg_learner',
                objects_to_save={
                    'counter': self._counter,
                    'policy': self._policy_networks,
                    'critic': self._critic_networks,
                    'observation': self._observation_networks,
                    'target_policy': self._target_policy_networks,
                    'target_critic': self._target_critic_networks,
                    'target_observation': self._target_observation_networks,
                    'policy_optimizer': self._policy_optimizers,
                    'critic_optimizer': self._critic_optimizers,
                    'num_steps': self._num_steps,
                })
            object_to_save = dict()
            for i in range(len(self._policy_networks)):
                object_to_save[f'policy_{i}'] = self._policy_networks[i]
                object_to_save[f'critic_mean_{i}'] = snt.Sequential([self._critic_networks[i], acme_nets.StochasticMeanHead()])
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save=object_to_save)

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

        self._edge_number = edge_number
        self._edge_action_size = edge_action_size

    @tf.function
    def _step(self, sample) -> Dict[str, tf.Tensor]:
        transitions: types.Transition = sample.data  # Assuming ReverbSample.
        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=tf.float64)

        with GradientTape(persistent=True) as tape:
            """Compute the loss for the policy and critic of edge nodes."""
            critic_losses = [[] for _ in range(self._edge_number)]
            policy_losses = [[] for _ in range(self._edge_number)]
            """Deal with the observations."""
            # the shpae of the transitions.observation is [batch_size, edge_number, edge_observation_size]
            batch_size = transitions.observation.shape[0]
            
            # myapp.debug(f"observation: {np.array(transitions.observation)}")
            
            # NOTE: the input of the edge_observation_network is 
            # [batch_size, edge_observation_size]
            # a_t_list = []
            # for i in range(len(self._target_observation_networks)):
            #     observation = transitions.observation[:, i, :]
            #     o_t = self._target_observation_networks[i](observation)
            #     o_t = tree.map_structure(tf.stop_gradient, o_t)
            #     a_t = self._target_policy_networks[i](o_t)
            #     a_t_list.append(a_t)
            
            # edge_a_t = tf.concat([a_t_list[i] for i in range(len(self._target_observation_networks))], axis=1)
            
            a_t_list = []
            for i in range(len(self._target_observation_networks)):
                observation = transitions.next_observation[:, i, :]
                o_t = self._target_observation_networks[i](observation)
                o_t = tree.map_structure(tf.stop_gradient, o_t)
                a_t = self._target_policy_networks[i](o_t)
                a_t_list.append(a_t)
            
            edge_next_a_t = tf.concat([a_t_list[i] for i in range(len(self._target_observation_networks))], axis=1)
            
            
            for edge_index in range(self._edge_number):

                o_tm1 = self._observation_networks[edge_index](
                    transitions.observation[:, edge_index, :])
                
                o_t = self._target_observation_networks[edge_index](
                    transitions.next_observation[:, edge_index, :])
                
                o_t = tree.map_structure(tf.stop_gradient, o_t)

                # Critic learning.
                q_tm1 = self._critic_networks[edge_index](o_tm1, tf.reshape(transitions.action, shape=[batch_size, -1]))
                q_t = self._target_critic_networks[edge_index](o_t, tf.reshape(edge_next_a_t, shape=[batch_size, -1]))

                # Critic loss.
                rewards = transitions.reward[:, edge_index]
                for i in range(self._edge_number):
                    rewards[i] = transitions.reward[:, -1]
                critic_loss = losses.categorical(q_tm1, rewards,
                                                discount * transitions.discount, q_t)
                
                # myapp.debug(f"edge_index: {edge_index}")
                # myapp.debug(f"critic_loss: {np.array(critic_loss)}")
                
                critic_losses[edge_index].append(critic_loss)

                # Actor learning
                if edge_index == 0:
                    dpg_a_t = self._policy_networks[edge_index](o_t)
                else:
                    dpg_a_t = tf.reshape(edge_next_a_t, shape=[batch_size, self._edge_number, self._edge_action_size])[:, 0, :]
                for i in range(self._edge_number):
                    if i != 0 and i != edge_index:
                        dpg_a_t = tf.concat([dpg_a_t, tf.reshape(edge_next_a_t, shape=[batch_size, self._edge_number, self._edge_action_size])[:, i, :]], axis=1)
                    elif i != 0 and i == edge_index:
                        dpg_a_t = tf.concat([dpg_a_t, self._policy_networks[edge_index](o_t)], axis=1)
                
                dpg_z_t = self._critic_networks[edge_index](o_t, dpg_a_t)
                dpg_q_t = dpg_z_t.mean()

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._clipping else None
                # myapp.debug(f"dpg_q_t: {np.array(dpg_q_t)}")
                # myapp.debug(f"dpg_a_t: {np.array(dpg_a_t)}")
                policy_loss = losses.dpg(
                    dpg_q_t,
                    dpg_a_t,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping)
                policy_losses[edge_index].append(policy_loss)
                
                # myapp.debug(f"policy_loss: {np.array(policy_loss)}")

            
            
            new_critic_losses = []
            new_policy_losses = []
            
            for i in range(self._edge_number):
                new_critic_losses.append(tf.reduce_mean(tf.stack(critic_losses[i], axis=0)))
                new_policy_losses.append(tf.reduce_mean(tf.stack(policy_losses[i], axis=0)))
            
            # for i in range(self._edge_number):
            #     myapp.debug(f"new_critic_losses {i}: {np.array(new_critic_losses[i])}")
            #     myapp.debug(f"new_policy_losses {i}: {np.array(new_policy_losses[i])}")
        
        # Get trainable variables.
        policy_variables = [self._policy_networks[i].trainable_variables for i in range(self._edge_number)]
        critic_variables = [(
            self._observation_networks[i].trainable_variables + self._critic_networks[i].trainable_variables
        ) for i in range(self._edge_number)]
        
        # Compute gradients.
        replica_context = tf.distribute.get_replica_context()
        
        policy_gradients =  [average_gradients_across_replicas(
            replica_context,
            tape.gradient(new_policy_losses[edge_index], policy_variables[edge_index])) for edge_index in range(self._edge_number)]
        critic_gradients =  [average_gradients_across_replicas(
            replica_context,
            tape.gradient(new_critic_losses[edge_index], critic_variables[edge_index])) for edge_index in range(self._edge_number)]
        
        # for edge_index in range(self._edge_number):
        
        #     myapp.debug(f"policy_gradients {edge_index}: {np.array(policy_gradients[edge_index])}")
        #     myapp.debug(f"critic_gradients {edge_index}: {np.array(critic_gradients[edge_index])}")
        
        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients = [tf.clip_by_global_norm(policy_gradient, 40.)[0] for policy_gradient in policy_gradients]
            critic_gradients = [tf.clip_by_global_norm(critic_gradient, 40.)[0] for critic_gradient in critic_gradients]
            
        # for edge_index in range(self._edge_number):
        #     myapp.debug(f"policy_gradients {edge_index}: {np.array(policy_gradients[edge_index])}")
        #     myapp.debug(f"critic_gradients {edge_index}: {np.array(critic_gradients[edge_index])}")
        # Apply gradients.
        for edge_index in range(self._edge_number):
            self._policy_optimizers[edge_index].apply(
                policy_gradients[edge_index], policy_variables[edge_index])
            self._critic_optimizers[edge_index].apply(
                critic_gradients[edge_index], critic_variables[edge_index])
        # Losses to track.
        object_to_return = dict()
        for edge_index in range(self._edge_number):
            object_to_return['policy_loss_' + str(edge_index)] = new_policy_losses[edge_index]
            object_to_return['critic_loss_' + str(edge_index)] = new_critic_losses[edge_index]
        
        return object_to_return

    @tf.function
    def _replicated_step(self):
        # Update target network
        online_variables = [(
            *self._observation_networks[i].variables,
            *self._critic_networks[i].variables,
            *self._policy_networks[i].variables,
        ) for i in range(len(self._observation_networks))]
        
        target_variables = [(
            *self._target_observation_networks[i].variables,
            *self._target_critic_networks[i].variables,
            *self._target_policy_networks[i].variables,
        ) for i in range(len(self._target_observation_networks))]
        
        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for online, target in zip(online_variables, target_variables):
                for src, dest in zip(online, target):
                    dest.assign(src)
        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        sample = next(self._iterator)

        # This mirrors the structure of the fetches returned by self._step(),
        # but the Tensors are replaced with replicated Tensors, one per accelerator.
        replicated_fetches = self._replicator.run(self._step, args=(sample,))

        # print("replicated_fetches: ", replicated_fetches)
        
        def reduce_mean_over_replicas(replicated_value):
            """Averages a replicated_value across replicas."""
            # The "axis=None" arg means reduce across replicas, not internal axes.
            return self._replicator.reduce(
                reduce_op=tf.distribute.ReduceOp.MEAN,
                value=replicated_value,
                axis=None)

        fetches = tree.map_structure(reduce_mean_over_replicas, replicated_fetches)

        return fetches

    def step(self):
        # Run the learning step.
        fetches = self._replicated_step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp
        
        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
            self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]


def get_first_available_accelerator_type(
    wishlist: Sequence[str] = ('TPU', 'GPU', 'CPU')) -> str:
    """Returns the first available accelerator type listed in a wishlist.
    Args:
        wishlist: A sequence of elements from {'CPU', 'GPU', 'TPU'}, listed in
        order of descending preference.
    Returns:
        The first available accelerator type from `wishlist`.
    Raises:
        RuntimeError: Thrown if no accelerators from the `wishlist` are found.
    """
    get_visible_devices = tf.config.get_visible_devices

    for wishlist_device in wishlist:
        devices = get_visible_devices(device_type=wishlist_device)
        if devices:
            return wishlist_device

    available = ', '.join(
        sorted(frozenset([d.type for d in get_visible_devices()])))
    raise RuntimeError(
        'Couldn\'t find any devices from {wishlist}.' +
        f'Only the following types are available: {available}.')


def average_gradients_across_replicas(replica_context, gradients):
    """Computes the average gradient across replicas.
    This computes the gradient locally on this device, then copies over the
    gradients computed on the other replicas, and takes the average across
    replicas.
    This is faster than copying the gradients from TPU to CPU, and averaging
    them on the CPU (which is what we do for the losses/fetches).
    Args:
        replica_context: the return value of `tf.distribute.get_replica_context()`.
        gradients: The output of tape.gradients(loss, variables)
    Returns:
        A list of (d_loss/d_varabiable)s.
    """

    # We must remove any Nones from gradients before passing them to all_reduce.
    # Nones occur when you call tape.gradient(loss, variables) with some
    # variables that don't affect the loss.
    # See: https://github.com/tensorflow/tensorflow/issues/783

    gradients_without_nones = [g for g in gradients if g is not None]
    original_indices = [i for i, g in enumerate(gradients) if g is not None]

    results_without_nones = replica_context.all_reduce('mean',
                                                        gradients_without_nones)
    results = [None] * len(gradients)
    for ii, result in zip(original_indices, results_without_nones):
        results[ii] = result

    return results 