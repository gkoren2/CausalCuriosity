"""
Cross-entropy planner for pybullet-finger envs.
Uses Uniform distributions instead of Gaussians.
"""
from CausalWorld.causal_world.task_generators.task import task_generator
from CausalWorld.causal_world.envs.causalworld import CausalWorld

from cem.frameskip_wrapper import FrameSkip
from scipy.spatial.distance import euclidean
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from collections import defaultdict, Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from loguru import logger
from tqdm import tqdm

from cem.diagonal_truncnorm import DiagonalTruncNorm
from cem.time_adaptive_uniform_bounds import TimeAdaptiveUniformBounds
from cem.uniform_bounds import UniformBounds
from tslearn.metrics import cdist_dtw
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
class CEMPlanner:
    def __init__(self, n_plans, horizon, action_space, sampler,
                 n_iterations, n_elite, cache_k,
                 warm_starts=False, warm_start_relaxation=0.0,
                 plan_action_repeat=1,
                 action_transformation=None,
                 rng=None,
                 viz_progress=False
                 ):

        # plan_action_repeat: independently of the environment frame-skip,
        # repeat actions in the plan this many times
        self.viz_progress = viz_progress
        self.plan_action_repeat = plan_action_repeat
        self.n_plans = n_plans
        self.horizon = horizon
        self.n_iterations = n_iterations
        self.n_elite = n_elite
        self.action_space = action_space
        self.action_transformation = action_transformation

        self.cache_k = cache_k
        self.plan_cache = defaultdict(list)
        self.warm_starts = warm_starts
        self.warm_start_relaxation = warm_start_relaxation
        self.sampler = sampler
        if sampler == 'uniform':
            self.action_dist = UniformBounds(relax=self.warm_start_relaxation)
        elif sampler == 'ta_uniform':
            self.action_dist = TimeAdaptiveUniformBounds(relax=self.warm_start_relaxation)
        elif sampler == 'diag_normal':
            assert not self.warm_starts
            assert self.warm_start_relaxation == 0.0
            self.action_dist = DiagonalTruncNorm(action_space=self.action_space)
        else:
            raise ValueError(f'Unknown sampler: {sampler}')

        self.action_dist.init_from_action_space(self.action_space, self.horizon)

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def plan(self, envs, save_state, restore_state):
        """
        :return: The action that was determined to be the best
        """
        if not self.warm_starts:
            print('init from action space')
            self.action_dist.init_from_action_space(self.action_space, self.horizon)
            print('lower', self.action_dist.lower)
            print('upper', self.action_dist.upper)
        else:
            assert self.warm_starts
            self.action_dist.shift_t(1, action_space=self.action_space)

        best_action_plan = None
        best_return = -float('inf')
        best_reward_sequence = None

        _best_returns = []
        _worst_returns = []

        envs = []
        print(f'self.n_plans: {self.n_plans}')

        if self.viz_progress:
            maybe_progbar = partial(tqdm, desc='Planning')
        else:
            maybe_progbar = lambda x: x
        # line 7 in Algorithm 1 - for iteration m to M 
        for i_iteration in maybe_progbar(range(self.n_iterations)):
            # line 8 : sample n_plans from the CEM 
            action_plans, rel_duration_plans = self.action_dist.sample(self.n_plans, self.rng)

            effective_horizon = self.horizon * self.plan_action_repeat
            discrete_duration_plans = np.round(rel_duration_plans * effective_horizon).astype(int)

            reward_sequences = []

            # For analysis only
            _reward_microsequences = []
            reward_cluster = []

            # line 9 : run each of the plans on each of the environments (while creating them)
            for i_plan in range(self.n_plans):

                plan_actions = action_plans[i_plan]
                plan_durations = discrete_duration_plans[i_plan]

                masses = [0.1,0.2,0.3,0.4,0.5]
                sizes = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                shapes = ['cube', 'sphere']
                observations = np.zeros((len(masses)*len(sizes)*len(shapes), 198, 3))

                counter = 0
                for shape in shapes:

                    for i in range(len(masses)):
                        for j in range(len(sizes)):

                            if i_iteration == 0 and i_plan==0:
                                task = task_generator(task_generator_id="lifting", 
                                    tool_block_mass=masses[i],
                                        tool_block_shape = shape,
                                        tool_block_size = sizes[j],)
                                env = CausalWorld(task=task, skip_frame = 1, enable_visualization=False)
                               
                                envs.append(env)
                            else:
                                env = envs[counter]
                            count = 0
                            env.reset()

                            for i_step in range(self.horizon):
                                step_reward = 0.0
                                action = plan_actions[i_step]
                                action_repeat = plan_durations[i_step]
                                if self.action_transformation is not None:
                                    action = self.action_transformation(action)

                                for i_repeat in range(action_repeat):
                                    next_state, reward, _, _ = env.step(action)
                                    observations[counter,count,:] = next_state
                                    count = count + 1
                            counter = counter + 1
                            del(env)

                km_sdtw = TimeSeriesKMeans(n_clusters=2, metric="softdtw", max_iter=100,max_iter_barycenter=5,metric_params={"gamma": .5},random_state=0).fit(observations)#[:, :, 2]
                y = km_sdtw.predict(observations)
                if len(np.unique(y))==1:    # if all observations belong to the same cluster
                    distance = -0.99
                else:
                    distance = silhouette_score(observations, y, metric="dtw")
                reward_cluster.append(distance)
                length = len(km_sdtw.predict(observations))
                
                print(f'plan {i_plan+1}/{self.n_plans}: obs clusters for {shapes[0]}', km_sdtw.predict(observations)[:length//2] )
                print(f'observations clusters for {shapes[1]}', km_sdtw.predict(observations)[length//2:] )

            # Take elite samples
            print('reward_cluster',reward_cluster)
            plan_returns = np.array(reward_cluster)
            print('plan_returns', plan_returns.shape)
            print('action_plans', action_plans.shape)
            elite_idxs = np.argsort(-plan_returns)[:self.n_elite]
            elite_action_plans = action_plans[elite_idxs, :, :]
            elite_duration_plans = rel_duration_plans[elite_idxs, :]
            self.action_dist.fit_to(
                elite_action_plans, elite_duration_plans,
                action_space=self.action_space)
            print('lower', self.action_dist.lower)
            print('upper', self.action_dist.upper)

            if np.max(plan_returns) > best_return:
                best_return = np.max(plan_returns)
                best_idx = np.argmax(plan_returns)
                best_action_plan = action_plans[best_idx]
                best_rel_duration_plan = rel_duration_plans[best_idx]
        logger.info(f'best_return: {best_return}')
        logger.info(f'best_action_plan: {best_action_plan}')
        logger.info(f'best_rel_duration_plan: {best_rel_duration_plan}')
        best_plan_incl_repeats = [a for a in best_action_plan
                                  for _ in range(self.plan_action_repeat)]
        return best_plan_incl_repeats, plan_returns

if __name__ == '__main__':
    # Some manual tests...

    print('lol')
    #rng = np.random.RandomState(1234)
    # low = np.array([0., -1., 10.], dtype=np.float32)
    # high = np.array([1., 2., 100.], dtype=np.float32)
    # bounds = UniformBounds.from_action_space(spaces.Box(low=low, high=high),
    #                                          horizon=2)
    # print(f'bounds.lower: {bounds.lower}')
    # print(f'bounds.upper: {bounds.upper}')
    #
    # data = np.random.uniform(0, 1, (5, 2, 3))
    # data[:, 0, 0] *= 10
    # data[:, 1, 2] = data[:, 1, 2] + 10
    #
    # print(f'np.min(data, axis=0): {np.min(data, axis=0)}')
    # print(f'np.max(data, axis=0): {np.max(data, axis=0)}')
    #
    # bounds = UniformBounds.fit_to(data)
    # print(f'bounds: {bounds}')

    # bounds = UniformBounds.from_action_space(action_space=spaces.Box(-1, 1, (2,)), horizon=1)
    #
    # for _ in range(30):
    #     actions = bounds.sample(n_plans=4, rng=rng)
    #     bounds = UniformBounds.fit_to(actions)
    #     print(f'bounds: {bounds}')
