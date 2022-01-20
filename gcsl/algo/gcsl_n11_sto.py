import numpy as np
from rlutil.logging import logger

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
import pdb
import time
import tqdm
import os.path as osp
import copy
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False


class GCSL:
    """Goal-conditioned Supervised Learning (GCSL).

    Parameters:
        env: A gcsl.envs.goal_env.GoalEnv
        policy: The policy to be trained (likely from gcsl.algo.networks)
        replay_buffer: The replay buffer where data will be stored
        validation_buffer: If provided, then 20% of sampled trajectories will
            be stored in this buffer, and used to compute a validation loss
        max_timesteps: int, The number of timesteps to run GCSL for.
        max_path_length: int, The length of each trajectory in timesteps

        # Exploration strategy

        explore_timesteps: int, The number of timesteps to explore randomly
        expl_noise: float, The noise to use for standard exploration (eps-greedy)

        # Evaluation / Logging Parameters

        goal_threshold: float, The distance at which a trajectory is considered
            a success. Only used for logging, and not the algorithm.
        eval_freq: int, The policy will be evaluated every k timesteps
        eval_episodes: int, The number of episodes to collect for evaluation.
        save_every_iteration: bool, If True, policy and buffer will be saved
            for every iteration. Use only if you have a lot of space.
        log_tensorboard: bool, If True, log Tensorboard results as well

        # Policy Optimization Parameters

        start_policy_timesteps: int, The number of timesteps after which
            GCSL will begin updating the policy
        batch_size: int, Batch size for GCSL updates
        n_accumulations: int, If desired batch size doesn't fit, use
            this many passes. Effective batch_size is n_acc * batch_size
        policy_updates_per_step: float, Perform this many gradient updates for
            every environment step. Can be fractional.
        train_policy_freq: int, How frequently to actually do the gradient updates.
            Number of gradient updates is dictated by `policy_updates_per_step`
            but when these updates are done is controlled by train_policy_freq
        lr: float, Learning rate for Adam.
        demonstration_kwargs: Arguments specifying pretraining with demos.
            See GCSL.pretrain_demos for exact details of parameters
    """

    def __init__(self,
                 env,
                 policy,
                 replay_buffer,
                 validation_buffer=None,
                 max_timesteps=1e6,
                 max_path_length=50,
                 # Exploration Strategy
                 explore_timesteps=1e4,
                 expl_noise=0.1,
                 # Evaluation / Logging
                 goal_threshold=0.05,
                 eval_freq=5e3,
                 eval_episodes=200,
                 save_every_iteration=False,
                 log_tensorboard=False,
                 # Policy Optimization Parameters
                 start_policy_timesteps=0,
                 batch_size=100,
                 n_accumulations=1,
                 policy_updates_per_step=1,
                 train_policy_freq=None,
                 demonstrations_kwargs=dict(),
                 lr=5e-4,
                 ):
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

        self.is_discrete_action = hasattr(self.env.action_space, 'n')

        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length

        self.explore_timesteps = explore_timesteps
        self.expl_noise = expl_noise

        self.goal_threshold = goal_threshold
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_every_iteration = save_every_iteration

        self.start_policy_timesteps = start_policy_timesteps

        if train_policy_freq is None:
            train_policy_freq = self.max_path_length

        self.train_policy_freq = train_policy_freq
        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.net.parameters(), lr=lr)
        self.m_policy_optimizer = torch.optim.Adam(self.policy.marg_net.parameters() , lr = 5e-4)
        #pdb.set_trace()
        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None
        self.imbalanced_goals = True
        self.imb_init_dist = False
        self.goal_side = 0
        self.init_side = 0
        self.is_offline = True

        if self.is_offline:
            self.replay_buffer.max_buffer_size = 50000

    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if self.is_discrete_action else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype)
        goals_torch = torch.tensor(goals, dtype=obs_dtype)
        actions_torch = torch.tensor(actions, dtype=action_dtype)
        horizons_torch = torch.tensor(horizons, dtype=obs_dtype)
        weights_torch = torch.tensor(weights, dtype=torch.float32)

        conditional_nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)

        nll = conditional_nll
        c_nll = self.policy.cond_loss
        m_nll = self.policy.marg_loss


        return torch.mean(nll * weights_torch) , torch.mean(c_nll * weights_torch) , torch.mean(m_nll * weights_torch)

    def sample_trajectory_buffer(self, greedy=False, noise=0, render=False):
        #Oth dimensison bias
        if self.imbalanced_goals:
            # print('Goal_Space_Low',self.env.goal_space.low.flatten()[0])
            # print('Goal_Space_High',self.env.goal_space.high.flatten()[0])
            g0_low = self.env.goal_space.low.flatten()[0]
            g0_high = self.env.goal_space.high.flatten()[0]
            g0_avg = (g0_high + g0_low) / 2
            left = np.random.rand() < 0.8
            goal_state = self.env.sample_goal()
            goal = self.env.extract_goal(goal_state)
            if left:
                # print('left')
                self.goal_side = -1
                while goal[0] > g0_avg:
                    goal_state = self.env.sample_goal()
                    goal = self.env.extract_goal(goal_state)
            else:
                # print('right')
                self.goal_side = 1
                while goal[0] <= g0_avg:
                    goal_state = self.env.sample_goal()
                    goal = self.env.extract_goal(goal_state)

            # print('goal_0', goal[0])

        else:
            # print('No Bias')
            self.goal_side = 0
            goal_state = self.env.sample_goal()
            goal = self.env.extract_goal(goal_state)

        states = []
        actions = []
        # print('goal_0_b',goal[0])
        if self.imb_init_dist:
            obs_low = self.env.observation_space.low.flatten()
            obs_high = self.env.observation_space.high.flatten()
            obs_avg = (obs_high + obs_low) / 2
            obs_avg_ = (obs_high - obs_low) / 2
            state = self.env.reset()
            left = np.random.rand() < 0.8
            if left:
                state[0] = np.random.rand() * obs_avg_[0] + obs_low[0]
                self.init_side = -1
            else:
                state[0] = obs_high[0] - np.random.rand() * obs_avg_[0]
                self.init_side = 1

            # update sampled  goal_state to a biased initial state
            goal_state[0] = state[0]
        else:
            state = self.env.reset()
        # print('goal_0_a',goal[0])
        for t in range(self.max_path_length):
            if render:
                self.env.render()

            states.append(state)

            observation = self.env.observation(state)
            horizon = np.arange(self.max_path_length) >= (
                        self.max_path_length - 1 - t)  # Temperature encoding of horizon
            action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy,
                                                noise=noise)[0]

            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            actions.append(action)
            state, _, _, _ = self.env.step(action)

        return np.stack(states), np.array(actions), goal_state

    def sample_trajectory(self, greedy=False, noise=0, render=False):

        goal_state = self.env.sample_goal()
        goal = self.env.extract_goal(goal_state)
        state = self.env.reset()
        if self.imbalanced_goals:
            g0_low = self.env.goal_space.low.flatten()[0]
            g0_high = self.env.goal_space.high.flatten()[0]
            g0_avg = (g0_high + g0_low) / 2
            if goal[0] < g0_avg:
                self.goal_side = -1
            elif goal[0] > g0_avg:
                self.goal_side = 1
            else:
                self.goal_side = 0
        elif self.imb_init_dist:
            obs_low = self.env.observation_space.low.flatten()
            obs_high = self.env.observation_space.high.flatten()
            obs_avg = (obs_high + obs_low) / 2
            if state[0] < obs_avg[0]:
                self.init_side = -1
            elif state[0] > obs_avg[0]:
                self.init_side = 1
            else:
                self.init_side = 0


        states = []
        actions = []


        for t in range(self.max_path_length):
            if render:
                self.env.render()

            states.append(state)

            observation = self.env.observation(state)
            horizon = np.arange(self.max_path_length) >= (
                        self.max_path_length - 1 - t)  # Temperature encoding of horizon
            action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy,
                                                noise=noise)[0]

            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            actions.append(action)
            state, _, _, _ = self.env.step(action)

        return np.stack(states), np.array(actions), goal_state

    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        avg_c_loss = 0
        avg_m_loss = 0

        self.policy_optimizer.zero_grad()
        self.m_policy_optimizer.zero_grad()
        for _ in range(self.n_accumulations):
            observations, actions, goals, _, horizons, weights = buffer.sample_batch(self.batch_size)
            loss , c_loss , m_loss = self.loss_fn(observations, goals, actions, horizons, weights)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
            if loss.isnan().any():
                pdb.set_trace()
            avg_loss += ptu.to_numpy(loss)
            avg_c_loss += ptu.to_numpy(c_loss)
            avg_m_loss += ptu.to_numpy(m_loss)

        self.policy_optimizer.step()
        self.m_policy_optimizer.step()
        ## Soft Update of target network
        self.policy.soft_update(tau = 0.01)

        return avg_loss / self.n_accumulations , avg_c_loss / self.n_accumulations , avg_m_loss / self.n_accumulations

    def validation_loss(self, buffer=None):
        if buffer is None:
            buffer = self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0

        avg_loss = 0
        avg_c_loss = 0
        avg_m_loss = 0
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(self.batch_size)
            loss, c_loss , m_loss = self.loss_fn(observations, goals, actions, horizons, weights)
            avg_loss += ptu.to_numpy(loss)
            avg_c_loss += ptu.to_numpy(c_loss)
            avg_m_loss += ptu.to_numpy(m_loss)
        return avg_loss / self.n_accumulations , avg_c_loss / self.n_accumulations , avg_m_loss / self.n_accumulations

    def pretrain_demos(self, demo_replay_buffer=None, demo_validation_replay_buffer=None, demo_train_steps=0):
        if demo_replay_buffer is None:
            return

        self.policy.train()
        with tqdm.trange(demo_train_steps) as looper:
            for _ in looper:
                loss , c_loss , m_loss = self.take_policy_step(buffer=demo_replay_buffer)
                validation_loss , vc_loss , vm_loss = self.validation_loss(buffer=demo_validation_replay_buffer)

                if running_loss is None:
                    running_loss = loss
                    running_loss_c = c_loss
                    running_loss_m = m_loss
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss
                    running_loss_c = 0.99 * running_loss_c + 0.01 * c_loss
                    running_loss_m = 0.99 * running_loss_m + 0.01 * m_loss
                if running_validation_loss is None:
                    running_validation_loss = validation_loss
                    running_validation_loss_c = vc_loss
                    running_validation_loss_m = vm_loss
                else:
                    running_validation_loss = 0.99 * running_validation_loss + 0.01 * validation_loss
                    running_validation_loss_c = 0.99 * running_validation_loss_c + 0.01 * vc_loss
                    running_validation_loss_m = 0.99*running_validation_loss_m + 0.01 * vm_loss

                looper.set_description('Loss: %.03f Validation Loss: %.03f' % (running_loss, running_validation_loss))

    def train(self):
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0
        timesteps_since_reset = 0

        iteration = 0
        running_loss = None
        running_validation_loss = None

        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(osp.join(logger.get_snapshot_dir(), 'tensorboard'))

        # Evaluation Code
        self.policy.eval()
        self.evaluate_policy(self.eval_episodes, total_timesteps=0, greedy=True, prefix='Eval')
        logger.record_tabular('policy loss', 0)
        logger.record_tabular('timesteps', total_timesteps)
        logger.record_tabular('epoch time (s)', time.time() - last_time)
        logger.record_tabular('total time (s)', time.time() - start_time)
        logger.record_tabular('Conditional loss', 0)
        logger.record_tabular('Marginal loss', 0)
        logger.record_tabular('Validation loss', 0)  # Handling None case
        logger.record_tabular('Conditional Validation loss', 0)
        logger.record_tabular('Marginal Validation loss', 0)
        last_time = time.time()
        logger.dump_tabular()
        # End Evaluation Code

        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:

                # Interact in environmenta according to exploration strategy.
                if self.is_offline:
                    if self.replay_buffer.current_buffer_size < self.replay_buffer.max_buffer_size:
                        if total_timesteps < self.explore_timesteps:
                            states, actions, goal_state = self.sample_trajectory_buffer(noise=1)
                        else:
                            states, actions, goal_state = self.sample_trajectory_buffer(greedy=False, noise=self.expl_noise)

                        # With some probability, put this new trajectory into the validation buffer
                        if self.validation_buffer is not None and np.random.rand() < 0.2:
                            self.validation_buffer.add_trajectory(states, actions, goal_state)
                        else:
                            self.replay_buffer.add_trajectory(states, actions, goal_state)
                else:
                    #print('Not online')
                    # Interact in environmenta according to exploration strategy.
                    if total_timesteps < self.explore_timesteps:
                        states, actions, goal_state = self.sample_trajectory_buffer(noise=1)
                    else:
                        states, actions, goal_state = self.sample_trajectory_buffer(greedy=False, noise=self.expl_noise)

                    if self.validation_buffer is not None and np.random.rand() < 0.2:
                        self.validation_buffer.add_trajectory(states, actions, goal_state)
                    else:
                        self.replay_buffer.add_trajectory(states, actions, goal_state)

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length

                ranger.update(self.max_path_length)

                # Take training steps
                if timesteps_since_train >= self.train_policy_freq and total_timesteps > self.start_policy_timesteps:
                    timesteps_since_train %= self.train_policy_freq
                    self.policy.train()
                    for _ in range(int(self.policy_updates_per_step * self.train_policy_freq)):
                        loss,c_loss,m_loss = self.take_policy_step()
                        validation_loss, vc_loss, vm_loss = self.validation_loss()
                        if running_loss is None:
                            running_loss = loss
                            running_loss_c = c_loss
                            running_loss_m = m_loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss
                            running_loss_c = 0.9 * running_loss_c + 0.1 * c_loss
                            running_loss_m = 0.9 * running_loss_m + 0.1 * m_loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                            running_validation_loss_c = vc_loss
                            running_validation_loss_m = vm_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss
                            running_validation_loss_c = 0.9 * running_validation_loss_c + 0.1 * vc_loss
                            running_validation_loss_m = 0.9 * running_validation_loss_m + 0.1 * vm_loss

                    self.policy.eval()
                    ranger.set_description('Loss: %s Validation Loss: %s' % (running_loss, running_validation_loss))

                    if self.summary_writer:
                        self.summary_writer.add_scalar('Losses/Train', running_loss, total_timesteps)
                        self.summary_writer.add_scalar('Losses/Validation', running_validation_loss, total_timesteps)

                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    iteration += 1
                    # Evaluation Code
                    self.policy.eval()
                    self.evaluate_policy(self.eval_episodes, total_timesteps=total_timesteps, greedy=True,
                                         prefix='Eval')
                    logger.record_tabular('policy loss', running_loss or 0)  # Handling None case
                    logger.record_tabular('Conditional loss', running_loss_c)
                    logger.record_tabular('Marginal loss',running_loss_m)
                    logger.record_tabular('Validation loss', running_validation_loss or 0)  # Handling None case
                    logger.record_tabular('Conditional Validation loss', running_validation_loss_c)
                    logger.record_tabular('Marginal Validation loss', running_validation_loss_m)
                    logger.record_tabular('timesteps', total_timesteps)
                    logger.record_tabular('epoch time (s)', time.time() - last_time)
                    logger.record_tabular('total time (s)', time.time() - start_time)
                    last_time = time.time()
                    logger.dump_tabular()

                    # Logging Code
                    if logger.get_snapshot_dir():
                        modifier = str(iteration) if self.save_every_iteration else ''
                        torch.save(
                            self.policy.state_dict(),
                            osp.join(logger.get_snapshot_dir(), 'policy%s.pkl' % modifier)
                        )
                        if hasattr(self.replay_buffer, 'state_dict'):
                            with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl' % modifier), 'wb') as f:
                                pickle.dump(self.replay_buffer.state_dict(), f)

                        full_dict = dict(env=self.env, policy=self.policy)
                        with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl' % modifier), 'wb') as f:
                            pickle.dump(full_dict, f)

                    ranger.reset()

    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0):
        env = self.env

        all_states = []
        all_goal_states = []
        all_actions = []
        goal_sides = np.zeros(eval_episodes)
        init_sides = np.zeros(eval_episodes)
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state = self.sample_trajectory(noise=0, greedy=greedy)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[-1], goal_state)
            init_sides[index] = self.init_side
            goal_sides[index] = self.goal_side
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)
        if self.imbalanced_goals:
            left = goal_sides < 0
            right = goal_sides > 0
            success_vec_left = success_vec[left]
            success_vec_right = success_vec[right]
            n_left = len(success_vec_left)
            n_right = len(success_vec_right)
        elif self.imb_init_dist:
            left = goal_sides < 0
            right = goal_sides > 0
            success_vec_left = success_vec[left]
            success_vec_right = success_vec[right]
            n_left = len(success_vec_left)
            n_right = len(success_vec_right)

        logger.record_tabular('%s num episodes' % prefix, eval_episodes)
        logger.record_tabular('%s avg final dist' % prefix, np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio' % prefix, np.mean(success_vec))
        if self.imbalanced_goals or self.imb_init_dist:
            logger.record_tabular('%s success ratio_left'%prefix, np.mean(success_vec_left))
            logger.record_tabular('%s success ratio_right' % prefix, np.mean(success_vec_right))
            logger.record_tabular('%s n_left' % prefix, n_left)
            logger.record_tabular('%s n_right' % prefix, n_right)

        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist' % prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio' % prefix, np.mean(success_vec), total_timesteps)
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s' % (prefix, key), value)

        return all_states, all_goal_states
