# AFZL-SAC: Adversarial Failure Zone Learning with Soft Actor-Critic
# Frejus Sutton's elegant modification to SAC for catastrophic failure avoidance

import os
import random
import time
from dataclasses import dataclass
import collections

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
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # AFZL specific arguments
    afzl_enabled: bool = True
    """enable Adversarial Failure Zone Learning"""
    failure_penalty_weight: float = 10.0
    """weight for failure zone penalty in Q-learning"""
    predictive_horizon: int = 3
    """steps ahead to check for potential failures"""
    failure_lr: float = 1e-3
    """learning rate for the failure predictor network"""
    target_failure_rate: float = 0.05
    """target failure rate for adaptive penalty weight"""
    adaptive_lambda: bool = True
    """whether to adapt the penalty weight to meet target failure rate"""
    lambda_lr: float = 1e-4
    """learning rate for the adaptive lambda multiplier"""

    # Evaluation arguments
    eval_on_finish: bool = False
    """if toggled, run a final evaluation after training"""
    eval_episodes: int = 10
    """number of evaluation episodes to run at the end"""
    eval_deterministic: bool = True
    """use the actor mean action during evaluation (deterministic)"""


class FailureNet(nn.Module):
    """Differentiable failure predictor network"""
    
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output logits for binary classification
        logits = self.fc3(x)
        # Return probability of failure
        return torch.sigmoid(logits).squeeze(-1)


class AdaptiveLambda:
    """Adaptive Lagrange multiplier for failure rate control"""
    
    def __init__(self, initial_value=10.0, target_rate=0.05, lr=1e-4, device='cpu'):
        self.lambda_param = torch.tensor(initial_value, device=device, requires_grad=True)
        self.target_rate = target_rate
        self.optimizer = optim.Adam([self.lambda_param], lr=lr)
        self.device = device
        
    def get_lambda(self):
        return torch.clamp(self.lambda_param, min=0.0, max=100.0)
    
    def update(self, current_failure_rate):
        """Update lambda to drive failure rate towards target"""
        # Convert to tensor for gradient computation
        current_rate_tensor = torch.tensor(current_failure_rate, dtype=torch.float32, device=self.device, requires_grad=False)
        target_rate_tensor = torch.tensor(self.target_rate, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # Loss: (current_rate - target_rate)^2
        loss = (current_rate_tensor - target_rate_tensor) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.get_lambda().item()


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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def is_failure_state(state, info, reward=None, done=None, env_step_info=None):
    """
    Determine if a state represents a failure
    """
    # If we have access to the environment's health status, use it
    if env_step_info is not None and hasattr(env_step_info, 'get'):
        is_unhealthy = env_step_info.get('is_unhealthy', False)
        if is_unhealthy:
            return True
    
    # Fallback: episode termination with poor reward indicates failure
    if done is not None and done and reward is not None:
        return reward < 0.0
        
    # Final fallback: extremely low torso height (definitely fallen)
    torso_height = state[0].item()
    return torso_height < 0.5


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # Initialize networks
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Initialize AFZL failure predictor
    state_dim = np.array(envs.single_observation_space.shape).prod()
    failure_net = FailureNet(state_dim).to(device) if args.afzl_enabled else None
    failure_optimizer = optim.Adam(failure_net.parameters(), lr=args.failure_lr) if args.afzl_enabled else None
    
    # Initialize adaptive lambda
    adaptive_lambda = AdaptiveLambda(
        initial_value=args.failure_penalty_weight,
        target_rate=args.target_failure_rate,
        lr=args.lambda_lr,
        device=device
    ) if args.afzl_enabled and args.adaptive_lambda else None

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
    start_time = time.time()

    # Training loop
    obs, _ = envs.reset(seed=args.seed)
    failure_count = 0
    recent_failure_rate = 0.0
    failure_rate_window = 1000  # Window for computing recent failure rate

    # Store recent transitions for failure prediction training
    recent_transitions = []
    max_recent_transitions = 10000  # Keep last 10k transitions for failure training

    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # AFZL: Detect failures for training the failure predictor
        failure_labels = []
        if args.afzl_enabled and failure_net is not None:
            for i in range(envs.num_envs):
                # Check if this is a failure state
                is_failure = (
                    terminations[i] and not truncations[i] and rewards[i] < 0
                ) or (len(obs[i]) > 0 and obs[i][0] < 0.4)  # torso height check
                
                failure_labels.append(is_failure)
                if is_failure:
                    failure_count += 1

        # Record episode statistics
        if "episode" in infos:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}, episodic_length={infos['episode']['l']}, failure_count={failure_count}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
            if args.afzl_enabled:
                writer.add_scalar("afzl/failure_count", failure_count, global_step)

        # Store transition in replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        
        # Store recent transitions for failure prediction training
        if args.afzl_enabled and failure_net is not None:
            for i in range(envs.num_envs):
                recent_transitions.append({
                    'obs': obs[i].copy(),
                    'failure_label': failure_labels[i] if i < len(failure_labels) else False
                })
            
            # Keep only recent transitions
            if len(recent_transitions) > max_recent_transitions:
                recent_transitions = recent_transitions[-max_recent_transitions:]
        
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # Train failure predictor if enabled
            if args.afzl_enabled and failure_net is not None and len(recent_transitions) > args.batch_size:
                # Sample from recent transitions for failure prediction training
                batch_indices = np.random.choice(len(recent_transitions), args.batch_size, replace=False)
                batch_transitions = [recent_transitions[i] for i in batch_indices]
                
                # Extract observations and failure labels
                batch_obs = torch.tensor(
                    [t['obs'] for t in batch_transitions], 
                    dtype=torch.float32, 
                    device=device
                )
                batch_failure_labels = torch.tensor(
                    [t['failure_label'] for t in batch_transitions], 
                    dtype=torch.float32, 
                    device=device
                )
                
                # Predict failure probabilities
                failure_probs = failure_net(batch_obs)
                
                # Binary cross-entropy loss for failure prediction
                failure_loss = F.binary_cross_entropy(failure_probs, batch_failure_labels)
                
                # Optimize failure predictor
                failure_optimizer.zero_grad()
                failure_loss.backward()
                failure_optimizer.step()
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                
                # AFZL: Add differentiable failure penalties to Q-targets
                if args.afzl_enabled and failure_net is not None:
                    # Get current lambda value
                    current_lambda = adaptive_lambda.get_lambda() if adaptive_lambda is not None else args.failure_penalty_weight
                    
                    # Predict failure probability for next states
                    next_failure_probs = failure_net(data.next_observations)
                    min_qf_next_target -= current_lambda * next_failure_probs.unsqueeze(1)
                
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Optimize Q-networks
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    
                    # AFZL: Add differentiable failure penalties to actor loss
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    if args.afzl_enabled and failure_net is not None:
                        current_lambda = adaptive_lambda.get_lambda() if adaptive_lambda is not None else args.failure_penalty_weight
                        current_failure_probs = failure_net(data.observations)
                        actor_loss += current_lambda * current_failure_probs.mean()

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

            # Update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Update adaptive lambda based on recent failure rate
            if args.afzl_enabled and adaptive_lambda is not None and global_step % 1000 == 0:
                # Compute recent failure rate (simplified - in practice you'd track this more carefully)
                recent_failure_rate = min(1.0, failure_count / max(1, global_step - args.learning_starts))
                new_lambda = adaptive_lambda.update(recent_failure_rate)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                
                if args.afzl_enabled and failure_net is not None:
                    if len(recent_transitions) > args.batch_size:
                        writer.add_scalar("afzl/failure_loss", failure_loss.item(), global_step)
                    writer.add_scalar("afzl/failure_rate", recent_failure_rate, global_step)
                    writer.add_scalar("afzl/recent_transitions", len(recent_transitions), global_step)
                    if adaptive_lambda is not None:
                        writer.add_scalar("afzl/lambda_value", adaptive_lambda.get_lambda().item(), global_step)
                    # Log current failure probability
                    current_failure_prob = failure_net(torch.tensor(obs, device=device, dtype=torch.float32)).mean().item()
                    writer.add_scalar("afzl/current_failure_prob", current_failure_prob, global_step)
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    
    # Final evaluation
    if args.eval_on_finish:
        eval_env = gym.make(args.env_id)
        returns = []
        lengths = []
        failures_in_eval = 0
        
        for ep in range(args.eval_episodes):
            obs, _ = eval_env.reset(seed=args.seed + 10 + ep)
            done = False
            truncated = False
            ep_return = 0.0
            ep_len = 0
            episode_had_failure = False
            
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
                
                # Check for failure in evaluation
                if args.afzl_enabled and failure_net is not None and not episode_had_failure:
                    if is_failure_state(obs_t.squeeze(0), {}, reward, done or truncated):
                        failures_in_eval += 1
                        episode_had_failure = True
                
                ep_return += float(reward)
                ep_len += 1
                
            returns.append(ep_return)
            lengths.append(ep_len)
        
        eval_env.close()
        
        # Log results
        writer.add_scalar("eval/return_mean", float(np.mean(returns)), args.total_timesteps)
        writer.add_scalar("eval/return_std", float(np.std(returns)), args.total_timesteps)
        writer.add_scalar("eval/return_min", float(np.min(returns)), args.total_timesteps)
        writer.add_scalar("eval/return_max", float(np.max(returns)), args.total_timesteps)
        writer.add_scalar("eval/length_mean", float(np.mean(lengths)), args.total_timesteps)
        if args.afzl_enabled:
            writer.add_scalar("eval/failures", failures_in_eval, args.total_timesteps)
            
        print("Final evaluation:")
        print(f"  Episodes: {args.eval_episodes}")
        print(f"  Return mean/std/min/max: {np.mean(returns):.2f} / {np.std(returns):.2f} / {np.min(returns):.2f} / {np.max(returns):.2f}")
        print(f"  Length mean/std: {np.mean(lengths):.1f} / {np.std(lengths):.1f}")
        if args.afzl_enabled:
            print(f"  Failures in evaluation: {failures_in_eval}/{args.eval_episodes}")
    
    writer.close()