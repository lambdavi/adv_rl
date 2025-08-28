import os
import random
import time
import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.buffer import ReplayBuffer


@dataclass
class Args:
    exp_name: str = "original_sac"
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
    env_id: str = "LunarLander-v3"
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
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

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


def main():
    parser = argparse.ArgumentParser(description='Train Original SAC on any gym environment')
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", 
                        help="Gym environment ID")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="Total timesteps for training")
    parser.add_argument("--learning-starts", type=int, default=5000,
                        help="Timesteps before learning starts")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--buffer-size", type=int, default=1000000,
                        help="Replay buffer size")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Use CUDA if available")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false",
                        help="Disable CUDA")
    parser.add_argument("--track", action="store_true",
                        help="Track with wandb")
    parser.add_argument("--capture-video", action="store_true",
                        help="Capture video of training")
    parser.add_argument("--render-test", action="store_true",
                        help="Render a test run after training")
    parser.add_argument("--use-failure-buffer", action="store_true",
                        help="Use failure buffer for danger zone clustering")
    parser.add_argument("--failure-penalty-weight", type=float, default=0.5,
                        help="Weight for failure penalty in reward shaping")
    parser.add_argument("--mixed-failure-fraction", type=float, default=0.3,
                        help="Fraction of minibatch sampled from failure buffer")
    parser.add_argument("--max-failure-clusters", type=int, default=20,
                        help="Maximum number of failure clusters to maintain")
    
    args = parser.parse_args()
    
    # Convert to dataclass format
    sac_args = Args(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
        cuda=args.cuda,
        track=args.track,
        capture_video=args.capture_video,
        use_failure_buffer=args.use_failure_buffer,
        failure_penalty_weight=args.failure_penalty_weight,
        mixed_failure_fraction=args.mixed_failure_fraction,
        max_failure_clusters=args.max_failure_clusters
    )
    
    run_name = f"{sac_args.env_id}__{sac_args.exp_name}__{sac_args.seed}__{int(time.time())}"
    if sac_args.track:
        import wandb

        wandb.init(
            project=sac_args.wandb_project_name,
            entity=sac_args.wandb_entity,
            sync_tensorboard=True,
            config=vars(sac_args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(sac_args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(sac_args.seed)
    np.random.seed(sac_args.seed)
    torch.manual_seed(sac_args.seed)
    torch.backends.cudnn.deterministic = sac_args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and sac_args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(sac_args.env_id, sac_args.seed + i, i, sac_args.capture_video, run_name) for i in range(sac_args.num_envs)]
    )
    
    # Check if environment supports continuous actions
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        print(f"Warning: {sac_args.env_id} does not have continuous action space.")
        print("Original SAC is designed for continuous actions. Consider using a different environment.")
        print("Supported environments include: Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4, Humanoid-v4")
        return

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=sac_args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=sac_args.policy_lr)

    # Automatic entropy tuning
    if sac_args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=sac_args.q_lr)
    else:
        alpha = sac_args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        sac_args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=sac_args.num_envs,
        handle_timeout_termination=False,
    )
    
    # Initialize reward tracking
    reward_history = []
    rolling_window = 100  # Track last 100 episodes
    reward_accumulator = 0  # Track accumulated rewards for periodic logging
    step_counter = 0  # Track steps since last logging
    
    start_time = time.time()

    print(f"🚀 Training Original SAC on {sac_args.env_id}!")
    print(f"💡 Environment: {sac_args.env_id}")
    print(f"💡 Device: {device}")
    print(f"💡 Total timesteps: {sac_args.total_timesteps}")
    print(f"💡 Learning starts at: {sac_args.learning_starts}")
    print(f"💡 Tensorboard logs: runs/{run_name}")

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=sac_args.seed)
    for global_step in range(sac_args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < sac_args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Accumulate rewards for periodic logging
        reward_accumulator += np.mean(rewards)
        step_counter += 1

        # Log immediate rewards every 100 steps
        if global_step % 100 == 0:
            writer.add_scalar("charts/immediate_reward", np.mean(rewards), global_step)
            writer.add_scalar("charts/step_reward_mean", np.mean(rewards), global_step)
            writer.add_scalar("charts/step_reward_std", np.std(rewards), global_step)
            
            # Log accumulated rewards every 1000 steps
            if step_counter >= 1000:
                avg_reward_per_step = reward_accumulator / step_counter
                writer.add_scalar("charts/avg_reward_per_step", avg_reward_per_step, global_step)
                writer.add_scalar("charts/accumulated_reward", reward_accumulator, global_step)
                reward_accumulator = 0
                step_counter = 0
                writer.flush()  # Ensure data is written to disk

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
                    if sac_args.env_id == "LunarLander-v3":
                        success = episodic_return >= 200  # LunarLander success threshold
                        writer.add_scalar("charts/success_rate", float(success), global_step)
                        if global_step % 1000 == 0:
                            print(f"  Success: {success}")
                    elif sac_args.env_id == "Hopper-v4":
                        success = episodic_return >= 1000  # Hopper success threshold
                        writer.add_scalar("charts/success_rate", float(success), global_step)
                        if global_step % 1000 == 0:
                            print(f"  Success: {success}")
                    
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                # Safety check: only access final_observation if it exists
                if "final_observation" in infos and idx < len(infos["final_observation"]):
                    real_next_obs[idx] = infos["final_observation"][idx]
                # If final_observation is not available, keep the current next_obs
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > sac_args.learning_starts:
            data = rb.sample(sac_args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * sac_args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % sac_args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    sac_args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if sac_args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % sac_args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(sac_args.tau * param.data + (1 - sac_args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(sac_args.tau * param.data + (1 - sac_args.tau) * target_param.data)

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
                if sac_args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
    
    # Optional: render a test run
    if args.render_test:
        print("\n🎬 Rendering test run...")
        env = gym.make(sac_args.env_id, render_mode='human')
        obs, _ = env.reset()
        
        for _ in range(1000):
            actions, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
            actions = actions.detach().cpu().numpy()
            obs, reward, done, truncated, _ = env.step(actions[0])
            done = done or truncated
            
            if done:
                obs, _ = env.reset()
        
        env.close()


if __name__ == "__main__":
    main()
