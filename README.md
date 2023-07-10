# Game-Theoretic-Deep-Reinforcement-Learning

This is the code of paper, named "Joint Task Offloading and Resource Optimization in NOMA-based Vehicular Edge Computing: A Game-Theoretic DRL Approach", and the proposed solution and comparison algorithms are implemented.

## Environment
The conda environment file is located in `environment.yml`.    
It can be used to create the environment by:    
```bash
conda env create -f environment.yml
```

## File Structure

### Main Function
The main() function of the repo is located in `Experiment/experiment.py`.

### Algorithms

- Multi-agent distributed distributional deep deterministic policy gradient (MAD4PG): `Experiment/run_mad4pg.py`
- Multi-agent deep deterministic policy gradient (MADDPG): `Experiment/run_maddpg.py`
- Distributed distributional deep deterministic policy gradient (D4PG): `Experiment/run_d4pg.py`
- Optimal resource allocation and task local processing only (ORL): `Experiment/run_optres_local.py`
- Optimal resource allocation and task migration only (ORM): `Experiment/run_optres_edge.py`

### Didi Dataset

The vehicular trajectories for November 16, 2016, generated in Chengdu and extracted from the Didi GAIA Open Data Set, can be found on [neardws/Vehicular-Trajectories-Processing-for-Didi-Open-Data](https://github.com/neardws/Vehicular-Trajectories-Processing-for-Didi-Open-Data).

## Citing this paper 
```bibtex
@article{xu2022joint,
  title={Joint task offloading and resource optimization in NOMA-based vehicular edge computing: A game-theoretic DRL approach},
  author={Xu, Xincao and Liu, Kai and Dai, Penglin and Jin, Feiyu and Ren, Hualing and Zhan, Choujun and Guo, Songtao},
  journal={Journal of Systems Architecture},
  pages={102780},
  year={2022},
  issn = {1383-7621},
  doi = {https://doi.org/10.1016/j.sysarc.2022.102780},
  url = {https://www.sciencedirect.com/science/article/pii/S138376212200265X},
  publisher={Elsevier}
}
```
