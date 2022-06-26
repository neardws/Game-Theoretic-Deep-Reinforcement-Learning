import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Environment.environment import make_environment_spec
from Agents.RANDOM.actors import RandomActor
from Experiment.make_environment import get_default_environment


def main(_):

    __, __, __, __, __, __, __, __, environment = get_default_environment()
    
    env_spec = make_environment_spec(environment)

    actor = RandomActor(
        spec=env_spec,
    )
    loop = EnvironmentLoop(environment, actor)
    loop.run(100)
