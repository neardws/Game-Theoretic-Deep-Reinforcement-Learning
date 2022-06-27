
import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from acme import types
import tree
import numpy as np
from Environment.environment import make_environment_spec
from Experiment.make_environment import get_default_environment

def generate_random_action(action_space: types.NestedSpec) -> types.NestedArray:
    """Generate a value from a potentially nested spec."""
    return tree.map_structure(lambda s: np.random.random(size=s.shape), action_space)

if __name__ == "__main__":
    __, __, __, __, __, __, __, __, environment = get_default_environment()
    
    spec = make_environment_spec(environment)
    
    random_action = generate_random_action(spec.actions)
    
    environment.compute_reward(action=random_action)
    
    print(environment._reward)