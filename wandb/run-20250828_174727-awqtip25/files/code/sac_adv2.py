# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils.buffer import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
    # Evaluation arguments
    eval_on_finish: bool = False
    """if toggled, run a final evaluation after training"""
    eval_episodes: int = 10
    """number of evaluation episodes to run at the end"""
    eval_deterministic: bool = True
    """use the actor mean action during evaluation (deterministic)"""
    
    # Failure buffer specific arguments
    use_failure_buffer: bool = False
    """whether to use failure buffer for danger zone clustering"""
    failure_penalty_weight: float = 0.5
    """weight for failure penalty in reward shaping"""
    mixed_failure_fraction: float = 0.3
    """fraction of minibatch sampled from failure buffer"""
    max_failure_clusters: int = 20
    """maximum number of failure clusters to maintain"""
    failure_distance_threshold: float = 0.5
    """distance threshold for clustering failure states"""



class FailureClusterBuffer:
    def __init__(self, max_clusters=20, distance_threshold=0.5):
        self.clusters = []  # List of (centroid, states, weight)
        self.max_clusters = max_clusters
        self.distance_threshold = distance_threshold
    
    def add_failure_state(self, state):
        state = np.array(state)
        
        # Find closest cluster
        min_distance = float('inf')
        closest_cluster_idx = -1
        
        for i, (centroid, states, weight) in enumerate(self.clusters):
            distance = np.linalg.norm(state - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_idx = i
        
        # If close enough to existing cluster, add to it
        if closest_cluster_idx >= 0 and min_distance < self.distance_threshold:
            centroid, states, weight = self.clusters[closest_cluster_idx]
            states.append(state)
            # Update centroid
            new_centroid = np.mean(states, axis=0)
            self.clusters[closest_cluster_idx] = (new_centroid, states, weight + 1)
        else:
            # Create new cluster
            if len(self.clusters) >= self.max_clusters:
                # Merge closest clusters
                self._merge_closest_clusters()
            self.clusters.append((state, [state], 1))
    
    def _merge_closest_clusters(self):
        if len(self.clusters) < 2:
            return
        
        min_distance = float('inf')
        merge_i, merge_j = 0, 1
        
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                distance = np.linalg.norm(self.clusters[i][0] - self.clusters[j][0])
                if distance < min_distance:
                    min_distance = distance
                    merge_i, merge_j = i, j
        
        # Merge clusters
        centroid1, states1, weight1 = self.clusters[merge_i]
        centroid2, states2, weight2 = self.clusters[merge_j]
        
        all_states = states1 + states2
        new_centroid = np.mean(all_states, axis=0)
        new_weight = weight1 + weight2
        
        # Remove old clusters and add merged one
        self.clusters.pop(max(merge_i, merge_j))
        self.clusters.pop(min(merge_i, merge_j))
        self.clusters.append((new_centroid, all_states, new_weight))
    
    def compute_penalty(self, states):
        if not self.clusters:
            return np.zeros(len(states))
        
        states_np = np.array(states)
        penalties = np.full(len(states_np), np.inf)
        for centroid, states_list, weight in self.clusters:
            distances = np.linalg.norm(states_np - centroid, axis=1)
            weighted = distances / (weight ** 0.5)
            penalties = np.minimum(penalties, weighted)
        return np.exp(-penalties)  # smaller distance = stronger penalty

    
    def sample_danger_states(self, n_samples):
        if not self.clusters:
            return []
        
        # Sample proportionally to cluster weights
        weights = [weight for _, _, weight in self.clusters]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return []
        
        sampled_states = []
        for _ in range(n_samples):
            # Choose cluster based on weight
            cluster_idx = np.random.choice(len(self.clusters), p=[w/total_weight for w in weights])
            centroid, states, weight = self.clusters[cluster_idx]
            
            # Sample from cluster with some noise
            if states:
                base_state = random.choice(states)
                noise = np.random.normal(0, 0.1, len(base_state))
                sampled_state = base_state + noise
                sampled_states.append(sampled_state)
        
        return sampled_states


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Print main hyperparameters at launch
    print("=" * 80)
    print("🚀 SAC HYPERPARAMETERS")
    print("=" * 80)
    print(f"Environment: {args.env_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Seed: {args.seed}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() and args.cuda else 'CPU'}")
    print()
    print("📊 Algorithm Parameters:")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning starts: {args.learning_starts:,}")
    print(f"  Gamma (discount): {args.gamma}")
    print(f"  Tau (target smoothing): {args.tau}")
    print(f"  Policy learning rate: {args.policy_lr}")
    print(f"  Q-learning rate: {args.q_lr}")
    print(f"  Policy frequency: {args.policy_frequency}")
    print(f"  Target network frequency: {args.target_network_frequency}")
    print(f"  Alpha (entropy): {args.alpha}")
    print(f"  Autotune alpha: {args.autotune}")
    print()
    print("⚠️  FAILURE BUFFER SETTINGS:")
    if args.use_failure_buffer:
        print(f"  ❌ FAILURE BUFFER: ENABLED")
        print(f"    Failure penalty weight: {args.failure_penalty_weight}")
        print(f"    Mixed failure fraction: {args.mixed_failure_fraction}")
        print(f"    Max failure clusters: {args.max_failure_clusters}")
        print(f"    Failure distance threshold: {args.failure_distance_threshold}")
    else:
        print(f"  ✅ FAILURE BUFFER: DISABLED (standard SAC)")
    print("=" * 80)
    print()
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    
    # Initialize failure buffer if enabled
    if args.use_failure_buffer:
        failure_buffer = FailureClusterBuffer(
            max_clusters=args.max_failure_clusters,
            distance_threshold=args.failure_distance_threshold
        )
    else:
        failure_buffer = None
    
    # Initialize reward tracking
    reward_history = []
    rolling_window = 100  # Track last 100 episodes
    
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    
    # Track trajectory for failure detection
    trajectory_buffer = [[] for _ in range(args.num_envs)]
    
    # Fallback episodic trackers (in case final_info is missing/rare)
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    
    # Step reward accumulators for periodic feedback
    step_reward_accumulator = 0.0
    step_counter = 0
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Fallback: accumulate per-episode stats regardless of final_info
        ep_returns += rewards
        ep_lengths += 1
        
        # Periodic step-level logging (every 100 steps)
        step_reward_accumulator += float(np.mean(rewards))
        step_counter += 1
        if global_step % 100 == 0:
            avg_step_reward = step_reward_accumulator / max(step_counter, 1)
            writer.add_scalar("charts/immediate_reward", float(np.mean(rewards)), global_step)
            writer.add_scalar("charts/avg_reward_per_step", avg_step_reward, global_step)
            step_reward_accumulator = 0.0
            step_counter = 0

        # Failure detection and buffer management
        if args.use_failure_buffer and failure_buffer is not None:
            for env_idx in range(args.num_envs):
                # Add current state to trajectory
                trajectory_buffer[env_idx].append(obs[env_idx])
                
                # Check for failure (negative reward or termination with negative reward)
                is_failure = rewards[env_idx] < -50 or (terminations[env_idx] and rewards[env_idx] < 0)
                
                if is_failure and len(trajectory_buffer[env_idx]) > 0:
                    # Add final states from trajectory to failure buffer
                    final_states = trajectory_buffer[env_idx][-5:]  # Last 5 states
                    for state in final_states:
                        failure_buffer.add_failure_state(state)
                
                # Clear trajectory if episode ends
                if terminations[env_idx] or truncations[env_idx]:
                    trajectory_buffer[env_idx] = []

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # Preferred path: use final_info emitted by RecordEpisodeStatistics
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    episodic_return = info['episode']['r']
                    episodic_length = info['episode']['l']
                    
                    # Track reward history for rolling statistics
                    reward_history.append(episodic_return)
                    if len(reward_history) > rolling_window:
                        reward_history.pop(0)
                    
                    # Calculate rolling statistics
                    if len(reward_history) > 0:
                        rolling_mean = np.mean(reward_history)
                        rolling_std = np.std(reward_history)
                        rolling_min = np.min(reward_history)
                        rolling_max = np.max(reward_history)
                    
                    # Print reward info during training
                    if global_step % 1000 == 0:  # Print every 1000 steps
                        print(f"Step {global_step}: Return={episodic_return:.2f}, Length={episodic_length}")
                        if len(reward_history) > 0:
                            print(f"  Rolling Avg (last {len(reward_history)}): {rolling_mean:.2f} ± {rolling_std:.2f}")
                            print(f"  Rolling Range: [{rolling_min:.2f}, {rolling_max:.2f}]")
                    
                    # Log to tensorboard
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                    
                    # Log rolling statistics
                    if len(reward_history) > 0:
                        writer.add_scalar("charts/rolling_mean_return", rolling_mean, global_step)
                        writer.add_scalar("charts/rolling_std_return", rolling_std, global_step)
                        writer.add_scalar("charts/rolling_min_return", rolling_min, global_step)
                        writer.add_scalar("charts/rolling_max_return", rolling_max, global_step)
                    
                    # Track success/failure for environments with clear success criteria
                    if args.env_id == "LunarLander-v3":
                        success = episodic_return >= 200  # LunarLander success threshold
                        writer.add_scalar("charts/success_rate", float(success), global_step)
                        if global_step % 1000 == 0:
                            print(f"  Success: {success}")
                    elif args.env_id == "Hopper-v4":
                        success = episodic_return >= 1000  # Hopper success threshold
                        writer.add_scalar("charts/success_rate", float(success), global_step)
                        if global_step % 1000 == 0:
                            print(f"  Success: {success}")
                    
                    break
        
        # Fallback path: if any env is done this step, log episode stats built from accumulators
        for env_idx in range(args.num_envs):
            if terminations[env_idx] or truncations[env_idx]:
                episodic_return = float(ep_returns[env_idx])
                episodic_length = int(ep_lengths[env_idx])
                
                reward_history.append(episodic_return)
                if len(reward_history) > rolling_window:
                    reward_history.pop(0)
                
                if len(reward_history) > 0:
                    rolling_mean = np.mean(reward_history)
                    rolling_std = np.std(reward_history)
                    rolling_min = np.min(reward_history)
                    rolling_max = np.max(reward_history)
                
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                if len(reward_history) > 0:
                    writer.add_scalar("charts/rolling_mean_return", rolling_mean, global_step)
                    writer.add_scalar("charts/rolling_std_return", rolling_std, global_step)
                    writer.add_scalar("charts/rolling_min_return", rolling_min, global_step)
                    writer.add_scalar("charts/rolling_max_return", rolling_max, global_step)
                
                if args.env_id == "LunarLander-v3":
                    success = episodic_return >= 200
                    writer.add_scalar("charts/success_rate", float(success), global_step)
                elif args.env_id == "Hopper-v4":
                    success = episodic_return >= 1000
                    writer.add_scalar("charts/success_rate", float(success), global_step)
                
                # Reset accumulators for this env
                ep_returns[env_idx] = 0.0
                ep_lengths[env_idx] = 0

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and infos.get("final_observation", None) is not None:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # Apply failure penalty to rewards if failure buffer is enabled
            if args.use_failure_buffer and failure_buffer is not None and failure_buffer.clusters:
                # Compute failure penalties for current states
                penalties = failure_buffer.compute_penalty(data.observations.cpu().numpy())
                penalties = torch.FloatTensor(penalties).to(device) * args.failure_penalty_weight
                
                # Apply penalty to rewards (reward shaping)
                shaped_rewards = data.rewards.flatten() - penalties
                
                # Log penalty statistics
                if global_step % 100 == 0:
                    writer.add_scalar("failure_buffer/avg_penalty", penalties.mean().item(), global_step)
                    writer.add_scalar("failure_buffer/num_clusters", len(failure_buffer.clusters), global_step)
            else:
                shaped_rewards = data.rewards.flatten()
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = shaped_rewards + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    # Standard SAC actor loss (already in your code)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()

                    # Failure buffer actor penalty
                    if len(failure_buffer.clusters) > 0:
                        with torch.no_grad():
                            danger_penalty = failure_buffer.compute_penalty(obs)
                        danger_penalty = torch.FloatTensor(danger_penalty).to(device)

                        # add penalty term
                        actor_loss += args.failure_penalty_weight * danger_penalty.mean()


                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    
    # Optional final evaluation
    if args.eval_on_finish:
        eval_env = gym.make(args.env_id)
        returns = []
        lengths = []
        for ep in range(args.eval_episodes):
            obs, _ = eval_env.reset(seed=args.seed + 10 + ep)
            done = False
            truncated = False
            ep_return = 0.0
            ep_len = 0
            while not (done or truncated):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if args.eval_deterministic:
                    with torch.no_grad():
                        mean, _ = actor(obs_t)
                        action = torch.tanh(mean) * actor.action_scale + actor.action_bias
                else:
                    with torch.no_grad():
                        action, _, _ = actor.get_action(obs_t)
                action_np = action.squeeze(0).detach().cpu().numpy()
                obs, reward, done, truncated, _ = eval_env.step(action_np)
                ep_return += float(reward)
                ep_len += 1
            returns.append(ep_return)
            lengths.append(ep_len)
        eval_env.close()
        # Log aggregated metrics
        writer.add_scalar("eval/return_mean", float(np.mean(returns)), args.total_timesteps)
        writer.add_scalar("eval/return_std", float(np.std(returns)), args.total_timesteps)
        writer.add_scalar("eval/return_min", float(np.min(returns)), args.total_timesteps)
        writer.add_scalar("eval/return_max", float(np.max(returns)), args.total_timesteps)
        writer.add_scalar("eval/length_mean", float(np.mean(lengths)), args.total_timesteps)
        writer.add_scalar("eval/length_std", float(np.std(lengths)), args.total_timesteps)
        print("Final evaluation:")
        print(f"  Episodes: {args.eval_episodes}")
        print(f"  Return mean/std/min/max: {np.mean(returns):.2f} / {np.std(returns):.2f} / {np.min(returns):.2f} / {np.max(returns):.2f}")
        print(f"  Length mean/std: {np.mean(lengths):.1f} / {np.std(lengths):.1f}")
    writer.close()