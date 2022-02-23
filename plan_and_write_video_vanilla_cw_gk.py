import random
import time
from datetime import datetime
from pathlib import Path
# from causal_world.task_generators.task import task_generator
# from causal_world.envs.causalworld import CausalWorld

import moviepy.editor as mpy
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from loguru import logger
from tqdm import tqdm

from cem_planner_vanilla_cw_gk import CEMPlanner

from cem.frameskip_wrapper import FrameSkip
from plan_action_spaces import get_plan_action_space


def main(output_dir, viz_progress=False):
    scenario = 'lift'

    if scenario == 'spin':      # what is 'spin' ?
        n_frames_per_episode = 198
    else:
        n_frames_per_episode = 198

    # Play around with these settings. They are not yet optimized
    total_budget = 400
    plan_horizon = 6
    n_plan_iterations = 20      # this is the M in Algorithm 1 - number of iterations 
    frame_skip = 1
    action_mode = 'RGB_asym'    # max degrees of freedom for the 3 fingers
    sampler = 'uniform'
    warm_starts = False
    warm_start_relaxation = 0.0
    elite_fraction = 0.1

    plan_action_repeat = n_frames_per_episode // plan_horizon       # 
    n_plan_cache_k = plan_horizon
    # n_plans to sample from the CEM  = total budget (=400)/n_iterations (20) = 20 plans per iteration 
    n_plans = total_budget * n_plan_cache_k // (plan_horizon * n_plan_iterations) 

    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))

    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1

    # do we need this here ? it runs without it...
    # bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)

    seed = 1235

    logger.info(f'seed: {seed}')
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    #obs = env.reset()
    episode_actions = []
    

    action_space, action_transformation = get_plan_action_space(action_mode)

    planner = CEMPlanner(n_plans=n_plans,       # num of plans to sample from CEM (line 8 in Algorithm 1)
                         horizon=plan_horizon,
                         action_space=action_space,
                         sampler=sampler,
                         n_iterations=n_plan_iterations,    # num of iteration (M in algorithm 1 line 7)
                         n_elite=n_plan_elite,
                         cache_k=n_plan_cache_k,        # what does this parameters mean ? 
                         warm_starts=warm_starts,
                         warm_start_relaxation=warm_start_relaxation,
                         plan_action_repeat=plan_action_repeat,
                         action_transformation=action_transformation,
                         rng=rng,
                         viz_progress=viz_progress)

    real_rewards = []

    frames = []

    actionSeq, plan_returns = planner.plan(None, None, None)
    print('actionSeq',len(actionSeq))
    print(actionSeq)
    print('plan_returns', plan_returns)

    # why do we need this line again ? 
    bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)

    try:
        for i_step in range(n_frames_per_episode):
            action = actionSeq[i_step]      #planner.plan(env, save_state, restore_state)
            env_action = action_transformation(action)
            # obs, reward, done, info = env.step(env_action)
            # real_rewards.append(reward)
            episode_actions.append(env_action)
            # rgb_array = env.render(mode='rgb_array')
            # frames.append((rgb_array * 255).astype(np.uint8))
    except KeyboardInterrupt:
        logger.info(f'Interrupted at step {i_step}')

    # logger.info(f'np.mean(real_rewards): {np.mean(real_rewards)}')

    print('Saving video...')
    # try:

    #     # clip = mpy.ImageSequenceClip(frames, fps=60)
    #     # video_path = str(Path(output_dir) / f'plan1.mp4')
    #     # clip.write_videofile(video_path)

    # except Exception as e:
    #     logger.info(e)

    print('Saving actions...')
    episode_actions = np.array(episode_actions)
    np.save(str(Path(output_dir) / 'episode_actions.npy'), episode_actions)
    print(str(Path(output_dir) / 'episode_actions.npy'))
    # bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    # env = FrameSkip(BlockManipulationEnv(bullet_client=bullet_client,
    #                                      scenario=scenario,
    #                                      control_rate_s=0.004,
    #                                              mass=0.1,
    #                                              size=0.0325,
    #                                              ),
    #                 frame_skip=frame_skip)
    # obs = env.reset()
    # episode_actions1 = []
    # frames1 = []
    # try:
    #     for i_step in range(n_frames_per_episode):
    #         action = actionSeq[i_step]#planner.plan(env, save_state, restore_state)
    #         env_action = action_transformation(action)
    #         obs, reward, done, info = env.step(env_action)
    #         #real_rewards.append(reward)
    #         episode_actions1.append(env_action)
    #         rgb_array = env.render(mode='rgb_array')
    #         frames1.append((rgb_array * 255).astype(np.uint8))
    # except KeyboardInterrupt:
    #     logger.info(f'Interrupted at step {i_step}')

    # logger.info(f'np.mean(real_rewards): {np.mean(real_rewards)}')

    # print('Saving video...')
    # try:
    #     clip1 = mpy.ImageSequenceClip(frames1, fps=60)
    #     video_path = str(Path(output_dir) / f'plan2.mp4')
    #     clip1.write_videofile(video_path)

    # except Exception as e:
    #     logger.info(e)

    #print('Saving actions...')
    #episode_actions = np.array(episode_actions)
    #np.save(str(Path(output_dir) / 'episode_actions.npy'), episode_actions)



if __name__ == "__main__":
    # print(f'SCENARIOS: {SCENARIOS}')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'./tmp/fingers_{timestamp}'
    Path(output_dir).mkdir()

    main(output_dir=output_dir, viz_progress=True)
