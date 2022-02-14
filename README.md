# Causal Curiosity - gk_main
Official implementation of Causal Curiosity: RL Agents Discovering Self-supervised Experiments for Causal Representation Learning at ICML 2021. 
[Paper](https://arxiv.org/abs/2010.03110) and [Website](https://sites.google.com/usc.edu/causal-curiosity/home)

This is the branch that I've created from the forked official main branch
## Installation

Download our version of [CausalWorld](https://github.com/rr-learning/CausalWorld) from this [drive](https://drive.google.com/drive/folders/1BWm0BuN8t3h9hJX-iA7Kp8q093Jub8fa?usp=sharing) link. Once downloaded add it to this repository and follow instructions to install [CausalWorld](https://github.com/rr-learning/CausalWorld).

You will also need [mujoco-py](https://github.com/openai/mujoco-py). Follow the installation instructions [here](https://github.com/openai/mujoco-py).
After installing mujoco-py, you will need to edit the done property for each of the mujoco agents property files. The ```done``` property needs to be set to ```False```. Otherwise the environment will stop simulating if the agents orientation exceeds a threshold. 

### Elaborate installation steps
- Clone this repo
- Download the CausalWorld framework and add it to the corresponding folder as instructed above
- follow the instructions in the CausalWorld folder README.md to install the library
  - before generating the conda env according to the `environment.yml` make sure to edit this file to change the name of the environment as you'd like. e.g. `cw2`
  - `conda env create -f environment.yml`
  - `conda activate cw2`
  - `(cw2) pip install -e .`
- install scipy: `conda install scipy`
- instal moviepy: `pip install moviepy`
- install loguru: `pip install loguru`
- install tslearn: `pip install tslearn`
- make the following changes in the code:
  - in `cem_planner_vanilla_cw.py` line 6 - add `cem` to the module path: `from cem.frameskip_wrapper import FrameSkip`
  - in `plan_and_write_video_vanilla_cw.py` line 17 - do the same change
- `mkdir tmp` so that the reports will be written there
- in `CausalWorld/causal_world/envs/causalworld.py` line 107 change the path to: `"../assets/robot_properties_fingers"`

Now it should be able to run: 
`python plan_and_write_video_vanilla_cw.py` 


## Usage
For Mujoco experiments, run 
```python
python pnw_mujoco.py
```

For CausalWorld experiments, run
```python
python plan_and_write_video_vanilla_cw.py
```
## Citation
```
@article{sontakke2020causal,
  title={Causal Curiosity: RL Agents Discovering Self-supervised Experiments for Causal Representation Learning},
  author={Sontakke, Sumedh A and Mehrjou, Arash and Itti, Laurent and Sch{\"o}lkopf, Bernhard},
  journal={arXiv preprint arXiv:2010.03110},
  year={2020}
}
```
