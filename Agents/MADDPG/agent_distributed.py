# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the DDPG agent class."""

import copy
from typing import Callable, Dict, Optional, List

import acme
from acme import specs
from Agents.MADDPG.agent import DDPGNetworks, DDPGConfig, DDPGBuilder, get_replicator
from acme.tf import savers as tf2_savers
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf
from Utilities.FileOperator import load_obj
from environment_loop import EnvironmentLoop

# Valid values of the "accelerator" argument.
_ACCELERATORS = ('CPU', 'GPU', 'TPU')


class DistributedDDPG:
    """Program definition for DDPG."""

    def __init__(
        self,
        agent_number: int,
        agent_action_size: int,
        environment_file: str,
        networks: List[DDPGNetworks],
        accelerator: Optional[str] = None,
        num_actors: int = 1,
        num_caches: int = 0,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 32.0,
        n_step: int = 1,
        sigma: float = 0.3,
        clipping: bool = True,
        discount: float = 0.99,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        target_update_period: int = 100,
        variable_update_period: int = 1000,
        max_actor_steps: Optional[int] = None,
        log_every: float = 10.0,
    ):
        self._agent_number = agent_number
        self._agent_action_size = agent_action_size
        
        if accelerator is not None and accelerator not in _ACCELERATORS:
            raise ValueError(f'Accelerator must be one of {_ACCELERATORS}, '
                        f'not "{accelerator}".')

        environment = load_obj(environment_file)
        if not environment_spec:
            environment_spec = specs.make_environment_spec(environment)

        self._environment_file = environment_file
        self._networks = networks
        self._environment_spec = environment_spec
        self._sigma = sigma
        self._num_actors = num_actors
        self._num_caches = num_caches
        self._max_actor_steps = max_actor_steps
        self._log_every = log_every
        self._accelerator = accelerator
        self._variable_update_period = variable_update_period

        self._builder = DDPGBuilder(
            # TODO(mwhoffman): pass the config dataclass in directly.
            # TODO(mwhoffman): use the limiter rather than the workaround below.
            DDPGConfig(
                accelerator=accelerator,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                variable_update_period=variable_update_period,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sigma=sigma,
                clipping=clipping,
            ))

    def replay(self):
        """The replay storage."""
        return self._builder.make_replay_tables(self._environment_spec)

    def counter(self):
        return tf2_savers.CheckpointingRunner(counting.Counter(),
                                            time_delta_minutes=1,
                                            subdirectory='counter')

    def coordinator(self, counter: counting.Counter):
        return lp_utils.StepsLimiter(counter, self._max_actor_steps)

    def learner(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ):
        """The Learning part of the agent."""

        # If we are running on multiple accelerator devices, this replicates
        # weights and updates across devices.
        replicator = get_replicator(self._accelerator)

        with replicator.scope():
            # Create the networks to optimize (online) and target networks.
            online_networks = self._networks
            target_networks = [copy.deepcopy(online_network) for online_network in online_networks]

            # Initialize the networks.
            for online_network, target_network in zip(online_networks, target_networks):
                online_network.init(self._environment_spec)
                target_network.init(self._environment_spec)

        dataset = self._builder.make_dataset_iterator(replay)

        counter = counting.Counter(counter, 'learner')
        logger = loggers.make_default_logger(
            'learner', time_delta=self._log_every, steps_key='learner_steps')

        return self._builder.make_learner(
            agent_number=self._agent_number,
            agent_action_size=self._agent_action_size,
            networks=(online_networks, target_networks),
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=True,
        )

    def actor(
        self,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> EnvironmentLoop:
        """The actor process."""

        # Create the behavior policy.
        networks = self._networks
        for network in networks:
            network.init(self._environment_spec)
        policy_networks = []
        for _ in range(self._agent_number):
            policy_network = networks[_].make_policy(
                environment_spec=self._environment_spec,
                sigma=self._sigma,
            )
            policy_networks.append(policy_network)

        # Create the agent.
        actor = self._builder.make_actor(
            agent_number=self._agent_number,
            agent_action_size=self._agent_action_size,
            policy_networks=policy_networks,
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
        )

        # Create the environment.
        environment = load_obj(self._environment_file)

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(counter, 'actor')
        logger = loggers.make_default_logger(
            'actor',
            save_data=False,
            time_delta=self._log_every,
            steps_key='actor_steps')

        # Create the loop to connect environment and agent.
        return EnvironmentLoop(environment, actor, counter, logger)

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        logger: Optional[loggers.Logger] = None,
    ):
        """The evaluation process."""

        # Create the behavior policy.
        networks = self._networks
        for network in networks:
            network.init(self._environment_spec)
        policy_networks = []
        for _ in range(self._agent_number):
            policy_network = networks[_].make_policy(self._environment_spec)
            policy_networks.append(policy_network)

        # Create the agent.
        actor = self._builder.make_actor(
            agent_number=self._agent_number,
            agent_action_size=self._agent_action_size,
            policy_networks=policy_networks,
            variable_source=variable_source,
        )

        # Make the environment.
        environment = load_obj(self._environment_file)

        # Create logger and counter.
        counter = counting.Counter(counter, 'evaluator')
        logger = logger or loggers.make_default_logger(
            'evaluator',
            time_delta=self._log_every,
            steps_key='evaluator_steps',
        )

        # Create the run loop and return it.
        return EnvironmentLoop(environment, actor, counter, logger)

    def build(self, name='ddpg'):
        """Build the distributed agent topology."""
        program = lp.Program(name=name)

        with program.group('replay'):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group('counter'):
            counter = program.add_node(lp.CourierNode(self.counter))

        if self._max_actor_steps:
            with program.group('coordinator'):
                _ = program.add_node(lp.CourierNode(self.coordinator, counter))

        with program.group('learner'):
            learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

        with program.group('evaluator'):
            program.add_node(lp.CourierNode(self.evaluator, learner, counter))

        if not self._num_caches:
            # Use our learner as a single variable source.
            sources = [learner]
        else:
            with program.group('cacher'):
                # Create a set of learner caches.
                sources = []
                for _ in range(self._num_caches):
                    cacher = program.add_node(
                        lp.CacherNode(
                            learner, refresh_interval_ms=2000, stale_after_ms=4000))
                sources.append(cacher)

        with program.group('actor'):
            # Add actors which pull round-robin from our variable sources.
            for actor_id in range(self._num_actors):
                source = sources[actor_id % len(sources)]
                program.add_node(lp.CourierNode(self.actor, replay, source, counter))

        return program