import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from absl import app
from Experiment import run_mad4pg

if __name__ == '__main__':
    app.run(run_mad4pg.main)
