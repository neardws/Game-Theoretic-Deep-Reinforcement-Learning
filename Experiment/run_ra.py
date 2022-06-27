import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from environment_loop import EnvironmentLoop
from Environment.environment import make_environment_spec
from Agents.RANDOM.actors import RandomActor
from Experiment.make_environment import get_default_environment
import numpy as np

def main(_):

    __, __, __, __, distance_matrix, __, __, __, environment = get_default_environment()
    
    # distance_matrix_shape = distance_matrix.shape
    
    # vehicle_numbers_within_edge = np.zeros((9, 300))
    
    # for t in range(distance_matrix_shape[2]):
    #     for e in range(distance_matrix_shape[1]):
    #         for v in range(distance_matrix_shape[0]):
    #             if distance_matrix[v, e, t] <= 500:
    #                 vehicle_numbers_within_edge[e, t] += 1
    
    # print("vehicle_numbers_within_edge: ", vehicle_numbers_within_edge)
    
    env_spec = make_environment_spec(environment)

    actor = RandomActor(
        spec=env_spec,
    )
    loop = EnvironmentLoop(environment, actor)
    loop.run(100)


if __name__ == "__main__":
    main(sys.argv)