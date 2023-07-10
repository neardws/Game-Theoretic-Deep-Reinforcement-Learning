from acme import core
from acme import specs
from acme import types
import dm_env
import tree
import numpy as np


class RandomActor(core.Actor):
    """Fake actor which generates random actions and validates specs."""

    def __init__(self, spec: specs.EnvironmentSpec):
        self._spec = spec
        self.num_updates = 0

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        _validate_spec(self._spec.observations, observation)
        return _generate_from_spec(self._spec.actions)

    def observe_first(self, timestep: dm_env.TimeStep):
        _validate_spec(self._spec.observations, timestep.observation)

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ):
        _validate_spec(self._spec.actions, action)
        _validate_spec(self._spec.rewards, next_timestep.reward)
        _validate_spec(self._spec.discounts, next_timestep.discount)
        _validate_spec(self._spec.observations, next_timestep.observation)

    def update(self, wait: bool = False):
        self.num_updates += 1

def _validate_spec(spec: types.NestedSpec, value: types.NestedArray):
    """Validate a value from a potentially nested spec."""
    tree.assert_same_structure(value, spec)
    tree.map_structure(lambda s, v: s.validate(v), spec, value)

def _generate_from_spec(spec: types.NestedSpec) -> types.NestedArray:
    """Generate a value from a potentially nested spec."""
    # print(tree.map_structure(lambda s: np.random.random(size=s.shape), spec))
    return tree.map_structure(lambda s: np.random.random(size=s.shape), spec)