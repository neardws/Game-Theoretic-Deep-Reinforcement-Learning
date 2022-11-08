# python3
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

"""A simple agent-environment training loop."""

import operator
import time
from typing import Optional, Sequence

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals

import dm_env
from dm_env import specs
import numpy as np
import tree
from Log.logger import myapp


class EnvironmentLoop(core.Worker):
    """A simple RL environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. Agent is updated if `should_update=True`. This can be used as:

        loop = EnvironmentLoop(environment, actor)
        loop.run(num_episodes)

    A `Counter` instance can optionally be given in order to maintain counts
    between different Acme components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.

    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger. A string `label` can be passed to easily
    change the label associated with the default logger; this is ignored if a
    `Logger` instance is given.

    A list of 'Observer' instances can be specified to generate additional metrics
    to be logged by the logger. They have access to the 'Environment' instance,
    the current timestep datastruct and the current action.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        actor: core.Actor,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        should_update: bool = True,
        label: str = 'environment_loop',
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self._should_update = should_update
        self._observers = observers

    def run_episode(self) -> loggers.LoggingData:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get an
        observation and then give that observation to the agent in order to retrieve
        an action.

        Returns:
        An instance of `loggers.LoggingData`.
        """
        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0
        
        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode.
        episode_return = tree.map_structure(_generate_zeros_from_spec,
                                            self._environment.reward_spec())
        timestep = self._environment.reset()
        # Make the first observation.
        self._actor.observe_first(timestep)
        for observer in self._observers:
        # Initialize the observer with the current state of the env after reset
        # and the initial timestep.
            observer.observe_first(self._environment, timestep)
        
        # Run an episode.
        e_time_start = time.time()
        select_action_time = 0
        environment_step_time = 0
        observer_and_update_time = 0
        
        
        cumulative_rewards: float = 0
        average_vehicle_SINRs: float = 0
        average_vehicle_intar_interferences: float = 0
        average_vehicle_inter_interferences: float = 0
        average_vehicle_interferences: float = 0
        average_transmision_times: float = 0
        average_wired_transmission_times: float = 0
        average_execution_times: float = 0
        average_service_times: float = 0 
        successful_serviced_numbers: float = 0
        task_required_numbers: float = 0
        task_offloaded_numbers: float = 0
        average_service_rate: float = 0
        average_offloaded_rate: float = 0
        average_local_rate: float = 0
        while not timestep.last():
        # Generate an action from the agent's policy and step the environment.
            # print("timestep.observation: ", timestep.observation[:, -2:])
            select_action_time_start = time.time()
            action = self._actor.select_action(timestep.observation)
            
            # print("action: ", action)
            select_action_time += time.time() - select_action_time_start
            
            environment_step_time_start = time.time()
            timestep, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
                average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number = self._environment.step(action)
            
            cumulative_rewards += cumulative_reward
            average_vehicle_SINRs += average_vehicle_SINR
            average_vehicle_intar_interferences += average_vehicle_intar_interference
            average_vehicle_inter_interferences += average_vehicle_inter_interference 
            average_vehicle_interferences += average_vehicle_interference
            average_transmision_times += average_transmision_time
            average_wired_transmission_times += average_wired_transmission_time
            average_execution_times += average_execution_time
            average_service_times += average_service_time
            successful_serviced_numbers += successful_serviced_number
            task_required_numbers += task_required_number
            task_offloaded_numbers += task_offloaded_number
            
            environment_step_time += time.time() - environment_step_time_start
            
            # myapp.debug(f"episode_steps: {episode_steps}")
            # myapp.debug(f"timestep.reward: {timestep.reward}")
            
            observer_and_update_time_start = time.time()
            
            # Have the agent observe the timestep and let the actor update itself.
            self._actor.observe(action, next_timestep=timestep)
            for observer in self._observers:
                # One environment step was completed. Observe the current state of the
                # environment, the current timestep and the action.
                observer.observe(self._environment, timestep, action)
            if self._should_update:
                self._actor.update()

            # Book-keeping.
            episode_steps += 1

            # Equivalent to: episode_return += timestep.reward
            # We capture the return value because if timestep.reward is a JAX
            # DeviceArray, episode_return will not be mutated in-place. (In all other
            # cases, the returned episode_return will be the same object as the
            # argument episode_return.)
            episode_return = tree.map_structure(operator.iadd,
                                                episode_return,
                                                timestep.reward)
            observer_and_update_time += time.time() - observer_and_update_time_start
        
        e_time_end = time.time()
        # print("episodes time taken: ", e_time_end - e_time_start)
        # print("select_action_time taken: ", select_action_time)
        # print("environment_step_time: ", environment_step_time)
        # print("observer_and_update_time: ", observer_and_update_time)
        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # if reward_1 > -5e4:
        #     for i in range(len(vehicle_transmission_times)):
        #         myapp.debug(f"i: {i}")
        #         myapp.debug(f"vehicle_transmission_times {i}: {vehicle_transmission_times[i]}")
        #         myapp.debug(f"vehicle_wired_transmission_times {i}: {vehicle_wired_transmission_times[i]}")
        #         myapp.debug(f"vehicle_execution_times {i}: {vehicle_execution_times[i]}")
        #         myapp.debug(f"rewards_1s {i}: {rewards_1s[i]}")
            # myapp.debug(f"timestep.reward: {timestep.reward}")
        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        average_vehicle_SINRs /= task_required_numbers
        average_vehicle_intar_interferences /= task_required_numbers
        average_vehicle_inter_interferences /= task_required_numbers 
        average_vehicle_interferences /= task_required_numbers
        
        average_transmision_times /= task_required_numbers
        average_wired_transmission_times /= task_required_numbers
        average_execution_times /= task_required_numbers
        average_service_times /= task_required_numbers
        
        # average_transmision_times /= successful_serviced_numbers
        # average_wired_transmission_times /= successful_serviced_numbers
        # average_execution_times /= successful_serviced_numbers
        # average_service_times /= successful_serviced_numbers
        average_service_rate = successful_serviced_numbers / task_required_numbers
        average_offloaded_rate = task_offloaded_numbers / task_required_numbers
        average_local_rate = (task_required_numbers - task_offloaded_numbers) / task_required_numbers
        # average_service_rate /= episode_steps
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
            'cumulative_reward': cumulative_rewards,
            'average_vehicle_SINRs': average_vehicle_SINRs,
            'average_vehicle_intar_interference': average_vehicle_intar_interferences,
            'average_vehicle_inter_interference': average_vehicle_inter_interferences,
            'average_vehicle_interferences': average_vehicle_interferences,
            'average_transmision_times': average_transmision_times,
            'average_wired_transmission_times': average_wired_transmission_times,
            'average_execution_times': average_execution_times,
            'average_service_times': average_service_times,
            'service_rate': average_service_rate,
            'offload_rate': average_offloaded_rate,
            'local_rate': average_local_rate,
        }
        result.update(counts)
        for observer in self._observers:
            result.update(observer.get_metrics())
        return result

    def run(self,
            num_episodes: Optional[int] = None,
            num_steps: Optional[int] = None):
        """Perform the run loop.

        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.

        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.

        Args:
        num_episodes: number of episodes to run the loop for.
        num_steps: minimal number of steps to run the loop for.

        Raises:
        ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """

        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return ((num_episodes is not None and episode_count >= num_episodes) or
                (num_steps is not None and step_count >= num_steps))

        episode_count, step_count = 0, 0
        with signals.runtime_terminator():
            while not should_terminate(episode_count, step_count):
                result = self.run_episode()
                episode_count += 1
                step_count += result['episode_length']
                # Log the given episode results.
                self._logger.write(result)

# Placeholder for an EnvironmentLoop alias


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)

