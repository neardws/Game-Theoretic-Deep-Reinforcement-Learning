import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from absl import app
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
memory_limit=4 * 1024
tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
tf.config.experimental.set_virtual_device_configuration(gpus[1], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

from Experiment import run_mad5pg
from Experiment import run_mad4pg
from Experiment import run_d4pg
from Experiment import run_ddpg
from Experiment import run_mpo
from Experiment import run_ra

if __name__ == '__main__':
    app.run(run_mad5pg.main) # /home/neardws/acme/40ab7856-f7c8-11ec-9987-04d9f5632a58/
    # app.run(run_mad4pg.main)
    # app.run(run_ddpg.main)
    # app.run(run_ra.main)