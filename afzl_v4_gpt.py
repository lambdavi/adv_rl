#!/usr/bin/env python3
# AFZL-SAC with learned FailureNet (single-file)
# Integrates a FailureNet into SAC: predicts p_fail(s) and penalizes actor and critic accordingly.

import os
import random
import time
from dataclasses import dataclass
import collections
from collections import deque
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Make sure your utils.buffer.ReplayBuffer is on PYTHONPATH
from utils.buffer import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "afzl"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "Hopper-v4"
    total_timesteps: int = 1_000_00
    num_envs: int = 1
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    # FailureNet / AFZL args
    afzl_enabled: bool = True
    failure_buffer_size: int = 2000
    failure_penalty_weight: float = 3.0
    failure_radius: float = 0.3  # not used in learned predictor; kept for compatibility
    failure_decay: float = 0.995  # not used in learned predictor; kept for compatibility
    predictive_horizon: int = 3
    penalty_update_frequency: int = 1  # frequency of applying penalties (in training steps)

    # FailureNet training hyperparams
    failure_lr: float = 1e-3
    failure_pos_frac: float = 0.25  # fraction of batch that should be positives (failures) when training FailureNet
    failure_train_every: int = 1  # train FailureNet every N gradient steps

    # Lagrangian option
    use_lagrangian: bool = False
    lagrangian_lr: float = 1e-3
    target_failure_rate: float = 0.02  # desired avg failure probability (per-step average under policy)
    lagrangian_init: float = 1.0
    lagrangian_clip: float = 100.0

    # Evaluation
    eval_on_finish: bool = False
    eval_episodes: int = 10
    eval_deterministic: bool = True


# -------------------------
# Networks
# -------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256, action_space=None):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, action_dim)
        self.fc_logstd = nn.Linear(hidden, action_dim)

        if action_space is None:
            self.register_buffer("action_scale", torch.tensor(1.0))
            self.register_buffer("action_bias", torch.tensor(0.0))
        else:
            high = action_space.high
            low = action_space.low
            self.register_buffer(
                "action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32)
            )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # log prob correction
        log_prob = normal.log_prob(x_t)
        # correction for tanh bijection
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class FailureNet(nn.Module):
    """Predict probability of failure within H steps."""
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, s):
        # Input s: [B, obs_dim]
        return self.net(s).view(-1)  # [B]


# -------------------------
# Lightweight Failure Buffer
# -------------------------
class FailureBuffer:
    """A simple circular buffer storing failure states (torch tensors)."""
    def __init__(self, capacity, state_dim, device):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((self.capacity, self.state_dim), device=self.device)

    def add(self, state):
        # state: torch tensor [state_dim]
        s = state.detach().to(self.device).view(-1)
        self.states[self.ptr] = s
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        if self.size == 0:
            return torch.zeros((0, self.state_dim), device=self.device)
        n = int(min(n, self.size))
        idx = np.random.randint(0, self.size, size=n)
        return self.states[idx]


# -------------------------
# Helpers
# -------------------------
def sample_states_from_rb(rb: ReplayBuffer, n: int):
    """
    Use rb.sample to get states. This assumes rb.sample(batch_size) returns a
    batch-like object with .observations (torch tensor [B,obs_dim]).
    We'll call rb.sample(n) and use its observations.
    """
    if n <= 0:
        return torch.zeros((0, rb.obs_shape_prod), dtype=torch.float32, device=next(iter(rb.__dict__.values())).device)
    data = rb.sample(int(n))
    # data.observations might already be a torch tensor on device
    states = data.observations
    if isinstance(states, np.ndarray):
        states = torch.tensor(states, dtype=torch.float32)
    return states


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env
    envs = gym.vector.SyncVectorEnv(
        [lambda idx=i: gym.make(args.env_id) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.array(envs.single_action_space.shape).prod())

    # Networks
    actor = Actor(obs_dim, action_dim, action_space=envs.single_action_space).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # FailureNet & buffer
    failure_net = FailureNet(obs_dim).to(device) if args.afzl_enabled else None
    failure_opt = optim.Adam(failure_net.parameters(), lr=args.failure_lr) if args.afzl_enabled else None
    failure_buffer = FailureBuffer(args.failure_buffer_size, obs_dim, device) if args.afzl_enabled else None

    # optional Lagrangian multiplier
    if args.use_lagrangian:
        lagrangian = torch.tensor(float(args.lagrangian_init), device=device, requires_grad=False)
    else:
        lagrangian = torch.tensor(float(args.failure_penalty_weight), device=device, requires_grad=False)

    # entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        alpha = log_alpha.exp().item()
    else:
        alpha = args.alpha

    # replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    # Per-env short histories for labeling predictive_horizon states on failure
    per_env_histories: List[deque] = [deque(maxlen=args.predictive_horizon) for _ in range(args.num_envs)]

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    episode_return = np.zeros(args.num_envs, dtype=float)
    episode_length = np.zeros(args.num_envs, dtype=int)
    failure_count = 0
    train_steps = 0
    avg_p_fail_running = 0.0
    avg_p_fail_count = 0

    for global_step in range(int(args.total_timesteps)):
        # select action
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                a_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                actions_t, _, _ = actor.get_action(a_tensor)
                actions = actions_t.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # update per-env short histories
        for i in range(args.num_envs):
            s_t = torch.tensor(obs[i], dtype=torch.float32, device=device)
            per_env_histories[i].append(s_t)

        # detect failures and add predictive_horizon states to FailureBuffer
        if args.afzl_enabled:
            for i in range(args.num_envs):
                is_unhealthy_termination = (terminations[i] and not truncations[i] and rewards[i] < 0)
                # fallback heuristic: large negative reward on termination or extremely low height
                definitely_fallen = False
                try:
                    torso_height = float(obs[i][0]) if len(obs[i]) > 0 else 1.0
                    definitely_fallen = torso_height < 0.4
                except Exception:
                    pass

                if is_unhealthy_termination or definitely_fallen:
                    # add the last H states as positives
                    for s in list(per_env_histories[i]):
                        failure_buffer.add(s)
                    failure_count += 1

        # store into replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # log episode stats
        for i, info in enumerate(infos):
            if isinstance(info, dict) and "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                print(f"global_step={global_step}, episodic_return={r:.1f}, episodic_length={l}, failure_count={failure_count}")
                writer.add_scalar("charts/episodic_return", r, global_step)
                writer.add_scalar("charts/episodic_length", l, global_step)
                if args.afzl_enabled:
                    writer.add_scalar("afzl/failure_count", failure_count, global_step)

        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)  # expects object with .observations, .next_observations, .actions, .rewards, .dones
            # ensure tensors are on device and in right dtype
            obs_batch = data.observations.to(device).float()
            actions_batch = data.actions.to(device).float()
            rewards_batch = data.rewards.to(device).float().view(-1)
            dones_batch = data.dones.to(device).float().view(-1)
            next_obs_batch = data.next_observations.to(device).float()

            # --- Train FailureNet (supervised) on mixed positives/negatives ---
            if args.afzl_enabled and train_steps % args.failure_train_every == 0:
                # positives from failure_buffer
                pos_n = int(max(1, args.failure_pos_frac * args.batch_size))
                pos = failure_buffer.sample(pos_n)
                # negatives: sample from replay (states that are not labeled failures)
                neg_n = args.batch_size - pos.shape[0]
                neg_data = rb.sample(neg_n)
                neg = neg_data.observations.to(device).float() if neg_data is not None else torch.zeros((0, obs_dim), device=device)
                # build training batch
                if pos.shape[0] == 0:
                    # no positives yet: skip training FailureNet
                    pass
                else:
                    x_fail = torch.cat([pos, neg], dim=0)
                    y_fail = torch.cat([torch.ones(pos.shape[0], device=device), torch.zeros(neg.shape[0], device=device)], dim=0)
                    # shuffle
                    idxs = torch.randperm(x_fail.shape[0], device=device)
                    x_fail = x_fail[idxs]
                    y_fail = y_fail[idxs]
                    failure_pred = failure_net(x_fail)
                    loss_failure = F.binary_cross_entropy(failure_pred, y_fail)
                    failure_opt.zero_grad()
                    loss_failure.backward()
                    failure_opt.step()
                    writer.add_scalar("afzl/failure_loss", loss_failure.item(), global_step)

            # --- Compute Q targets with failure penalty on next states ---
            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(next_obs_batch)
                qf1_next_target = qf1_target(next_obs_batch, next_actions)
                qf2_next_target = qf2_target(next_obs_batch, next_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi

                if args.afzl_enabled:
                    p_fail_next = failure_net(next_obs_batch)  # [B]
                    # detach predictor when applied to targets to avoid backprop through predictor
                    penalty_next = p_fail_next.detach() * (lagrangian.item() if not args.use_lagrangian else lagrangian.cpu().item())
                    # subtract immediate expected penalty (simple approach)
                    next_q_value = rewards_batch + (1 - dones_batch) * args.gamma * min_qf_next_target.view(-1) - penalty_next
                else:
                    next_q_value = rewards_batch + (1 - dones_batch) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(obs_batch, actions_batch).view(-1)
            qf2_a_values = qf2(obs_batch, actions_batch).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # policy (actor) update
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(obs_batch)
                    qf1_pi = qf1(obs_batch, pi)
                    qf2_pi = qf2(obs_batch, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

                    # base actor loss
                    actor_obj = (alpha * log_pi.view(-1) - min_qf_pi)

                    # failure penalty term for actor (we want per-sample penalty to push policy away)
                    if args.afzl_enabled:
                        p_fail_curr = failure_net(obs_batch).detach()  # detach predictor
                        # multiply penalty by lagrangian (learned or fixed)
                        penalty_curr = (lagrangian.item() if not args.use_lagrangian else lagrangian.cpu().item()) * p_fail_curr
                        actor_loss = (actor_obj + penalty_curr).mean()
                    else:
                        actor_loss = actor_obj.mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # alpha autotune
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi_det, _ = actor.get_action(obs_batch)
                        alpha_loss = (-log_alpha.exp() * (log_pi_det + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # soft-update targets
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Logging
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.afzl_enabled:
                    writer.add_scalar("afzl/buffer_size", failure_buffer.size, global_step)
                    # average p_fail on this batch
                    with torch.no_grad():
                        p_now = failure_net(obs_batch).mean().item()
                        writer.add_scalar("afzl/mean_p_fail_batch", p_now, global_step)
                        avg_p_fail_running += p_now
                        avg_p_fail_count += 1

                print("SPS:", int(global_step / (time.time() - start_time)))

            train_steps += 1

            # Update Lagrangian multiplier every so often (simple proportional step toward target failure rate)
            if args.afzl_enabled and args.use_lagrangian and train_steps % 50 == 0:
                if avg_p_fail_count > 0:
                    mean_p_fail = avg_p_fail_running / max(1, avg_p_fail_count)
                    # gradient-ascent style update for dual variable
                    lagrangian = lagrangian + args.lagrangian_lr * (mean_p_fail - args.target_failure_rate)
                    # clip to keep stable
                    lagrangian = torch.clamp(lagrangian, 0.0, args.lagrangian_clip)
                    writer.add_scalar("afzl/lagrangian", lagrangian.item(), global_step)
                    avg_p_fail_running = 0.0
                    avg_p_fail_count = 0

    envs.close()

    # Final evaluation
    if args.eval_on_finish:
        eval_env = gym.make(args.env_id)
        returns, lengths = [], []
        failures_in_eval = 0
        for ep in range(args.eval_episodes):
            ob, _ = eval_env.reset(seed=args.seed + 100 + ep)
            done = False
            truncated = False
            ep_ret = 0.0
            ep_len = 0
            while not (done or truncated):
                ot = torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                if args.eval_deterministic:
                    with torch.no_grad():
                        mean, _ = actor(ot)
                        action = torch.tanh(mean) * actor.action_scale + actor.action_bias
                else:
                    with torch.no_grad():
                        action, _, _ = actor.get_action(ot)
                action_np = action.squeeze(0).cpu().numpy()
                ob, r, done, truncated, info = eval_env.step(action_np)
                ep_ret += float(r)
                ep_len += 1
                # check failure
                if args.afzl_enabled and failure_buffer is not None:
                    is_fail = False
                    try:
                        if done and r < 0:
                            is_fail = True
                        elif ob[0] < 0.4:
                            is_fail = True
                    except Exception:
                        pass
                    if is_fail:
                        failures_in_eval += 1
            returns.append(ep_ret)
            lengths.append(ep_len)
        eval_env.close()
        writer.add_scalar("eval/return_mean", float(np.mean(returns)), args.total_timesteps)
        writer.add_scalar("eval/failures", failures_in_eval, args.total_timesteps)
        print("Final evaluation:")
        print(f"  Episodes: {args.eval_episodes}")
        print(f"  Return mean/std/min/max: {np.mean(returns):.2f} / {np.std(returns):.2f} / {np.min(returns):.2f} / {np.max(returns):.2f}")
        print(f"  Failures in evaluation: {failures_in_eval}/{args.eval_episodes}")

    writer.close()
