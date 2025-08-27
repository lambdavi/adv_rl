import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def sample_states(self, batch_size=None):
        """Sample just the states (for failure buffer)"""
        if batch_size is None:
            batch_size = min(len(self.buffer), 64)
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = [item[0] for item in batch]
        return np.array(states)
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output logits; probabilities via sigmoid for Bernoulli policy
        logits = self.fc3(x)
        return logits

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class FailureAwareSAC:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, use_failure_buffer=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.use_failure_buffer = use_failure_buffer
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)
        
        # Copy params to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Entropy temperature (alpha) with tuning
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = np.log(2.0)  # Bernoulli max entropy
        
        # Replay buffers - YOUR NOVEL DUAL BUFFER APPROACH!
        self.experience_buffer = ReplayBuffer(100000)
        self.failure_buffer = ReplayBuffer(10000)  # Separate buffer for failures
        
        # Adversarial parameters
        self.failure_penalty_weight = 0.5  # beta for reward shaping
        self.failure_distance_scale = 2.0   # Scale for distance computation
        self.mixed_failure_fraction = 0.3   # fraction of minibatch from failure buffer
        
        # Logging
        self.failure_penalties = []
        self.training_losses = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            probs = torch.sigmoid(logits)
            p = probs.cpu().numpy()[0][0]
        action = 1 if random.random() < p else 0
        return action, np.array([p], dtype=np.float32)
    
    def store_experience(self, state, action, reward, next_state, done):
        # Store in regular experience buffer
        action_continuous = np.array([action], dtype=np.float32)
        self.experience_buffer.push(state, action_continuous, reward, next_state, done)
        
        # Store failures separately (only if enabled)
        if self.use_failure_buffer and done and reward <= 0:
            self.failure_buffer.push(state, action_continuous, reward, next_state, done)
    
    def compute_failure_penalty(self, current_states):
        """Compute per-state penalties for proximity to failure states.
        Returns tensor of shape [batch, 1]."""
        batch_size = current_states.shape[0]
        if not self.use_failure_buffer or len(self.failure_buffer) < 10:
            penalties = torch.zeros((batch_size, 1), device=self.device)
            self.failure_penalties.append(0.0)
            return penalties
        
        # Sample failure states
        failure_states = self.failure_buffer.sample_states(min(64, len(self.failure_buffer)))
        failure_states = torch.FloatTensor(failure_states).to(self.device)
        
        # Compute distances between current states and failure states
        current_expanded = current_states.unsqueeze(1)  # [batch, 1, state_dim]
        failure_expanded = failure_states.unsqueeze(0)   # [1, failure_batch, state_dim]
        
        # L2 distance in state space
        distances = torch.norm(current_expanded - failure_expanded, dim=2, p=2)  # [batch, failure_batch]
        min_distances = torch.min(distances, dim=1)[0].unsqueeze(1)  # [batch,1]
        
        # Exponential penalty - stronger as you get closer to failure states
        penalties = self.failure_penalty_weight * torch.exp(-min_distances / self.failure_distance_scale)  # [batch,1]
        
        # Log average for analysis
        self.failure_penalties.append(penalties.mean().item())
        return penalties

    def sample_mixed_minibatch(self, batch_size):
        """Sample a mixed minibatch from experience and failure buffers."""
        if not self.use_failure_buffer or len(self.failure_buffer) == 0 or self.mixed_failure_fraction <= 0.0:
            return self.experience_buffer.sample(batch_size)
        num_fail_desired = int(batch_size * self.mixed_failure_fraction)
        num_fail_available = min(len(self.failure_buffer), num_fail_desired)
        num_exp_needed = batch_size - num_fail_available
        num_exp_available = min(len(self.experience_buffer), num_exp_needed)
        if num_exp_available < num_exp_needed:
            # fallback: reduce batch size if buffers are small
            batch_size = num_fail_available + num_exp_available
        
        fail_states = fail_actions = fail_rewards = fail_next_states = fail_dones = np.array([])
        if num_fail_available > 0:
            fs, fa, fr, fns, fd = self.failure_buffer.sample(num_fail_available)
            fail_states, fail_actions, fail_rewards, fail_next_states, fail_dones = fs, fa, fr, fns, fd
        
        if batch_size - num_fail_available > 0:
            es, ea, er, ens, ed = self.experience_buffer.sample(batch_size - num_fail_available)
            if num_fail_available > 0:
                states = np.concatenate([fail_states, es], axis=0)
                actions = np.concatenate([fail_actions, ea], axis=0)
                rewards = np.concatenate([fail_rewards, er], axis=0)
                next_states = np.concatenate([fail_next_states, ens], axis=0)
                dones = np.concatenate([fail_dones, ed], axis=0)
            else:
                states, actions, rewards, next_states, dones = es, ea, er, ens, ed
        else:
            states, actions, rewards, next_states, dones = fail_states, fail_actions, fail_rewards, fail_next_states, fail_dones
        
        # Shuffle the combined batch
        idx = np.arange(states.shape[0])
        np.random.shuffle(idx)
        return states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]
    
    def update(self, batch_size=64):
        if len(self.experience_buffer) < max(8, int(batch_size * (1 - (self.mixed_failure_fraction if self.use_failure_buffer else 0.0)))):
            return
        
        # Sample mixed minibatch
        states, actions, rewards, next_states, dones = self.sample_mixed_minibatch(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Reward shaping via failure proximity (detach to avoid gradients through reward)
        with torch.no_grad():
            penalties = self.compute_failure_penalty(states)
            shaped_rewards = rewards - penalties
        
        alpha = self.log_alpha.exp()

        # Critic update (discrete SAC with entropy in target)
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = torch.sigmoid(next_logits).clamp(1e-6, 1 - 1e-6)
            next_log_p = torch.log(next_probs)
            next_log_1mp = torch.log(1 - next_probs)

            a1 = torch.ones_like(next_probs)
            a0 = torch.zeros_like(next_probs)
            tq1_a1 = self.target_critic1(next_states, a1)
            tq2_a1 = self.target_critic2(next_states, a1)
            tq1_a0 = self.target_critic1(next_states, a0)
            tq2_a0 = self.target_critic2(next_states, a0)
            tmin_a1 = torch.min(tq1_a1, tq2_a1)
            tmin_a0 = torch.min(tq1_a0, tq2_a0)

            v_next = next_probs * (tmin_a1 - alpha * next_log_p) + (1 - next_probs) * (tmin_a0 - alpha * next_log_1mp)
            target_q = shaped_rewards + self.gamma * v_next * (1 - dones)
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor update: expectation over discrete actions with entropy
        logits = self.actor(states)
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        log_p = torch.log(probs)
        log_1mp = torch.log(1 - probs)

        a1 = torch.ones_like(probs)
        a0 = torch.zeros_like(probs)
        q1_a1 = self.critic1(states, a1)
        q2_a1 = self.critic2(states, a1)
        q1_a0 = self.critic1(states, a0)
        q2_a0 = self.critic2(states, a0)
        min_q_a1 = torch.min(q1_a1, q2_a1)
        min_q_a0 = torch.min(q1_a0, q2_a0)

        actor_loss = (probs * (alpha * log_p - min_q_a1) + (1 - probs) * (alpha * log_1mp - min_q_a0)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature tuning
        with torch.no_grad():
            entropy = -(probs * log_p + (1 - probs) * log_1mp)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log training info
        self.training_losses.append({
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item()
        })

def train_failure_aware_sac(use_failure_buffer=True):
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # Continuous representation of discrete action
    
    agent = FailureAwareSAC(state_dim, action_dim, use_failure_buffer=use_failure_buffer)
    
    episodes = 1000
    max_steps = 500
    episode_rewards = []
    failure_counts = []
    
    print("🚀 Training Failure-Aware SAC on CartPole!")
    mode = "ENABLED" if use_failure_buffer else "DISABLED"
    print(f"💡 Failure buffer is {mode} (reward shaping + mixed sampling)")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        failures_this_episode = 0
        
        for step in range(max_steps):
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Modified reward for CartPole failure detection
            if done and step < max_steps - 1:  # Early termination = failure
                reward = -1  # Failure penalty
                failures_this_episode += 1
            
            agent.store_experience(state, action, reward, next_state, done)
            
            if len(agent.experience_buffer) > 64:
                agent.update()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        failure_counts.append(failures_this_episode)
        
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            failure_buffer_size = len(agent.failure_buffer)
            recent_penalty = agent.failure_penalties[-1] if agent.failure_penalties else 0
            print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                  f"Failure Buffer: {failure_buffer_size}, "
                  f"Recent Failure Penalty: {recent_penalty:.4f}")
    
    return agent, episode_rewards, failure_counts

def plot_results(agent, episode_rewards, failure_counts):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0,0].plot(episode_rewards)
    axes[0,0].set_title('Episode Rewards Over Time')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Reward')
    axes[0,0].grid(True)
    
    # Failure penalties over time
    if agent.failure_penalties:
        axes[0,1].plot(agent.failure_penalties)
        axes[0,1].set_title('Failure Penalty Over Training Steps')
        axes[0,1].set_xlabel('Training Step')
        axes[0,1].set_ylabel('Failure Penalty')
        axes[0,1].grid(True)
    
    # Failure buffer growth
    axes[1,0].plot([len(agent.failure_buffer) for _ in range(len(episode_rewards))])
    axes[1,0].set_title('Failure Buffer Size')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Number of Stored Failures')
    axes[1,0].grid(True)
    
    # Training losses
    if agent.training_losses:
        actor_losses = [loss['actor_loss'] for loss in agent.training_losses[-500:]]
        failure_penalties = [loss['failure_penalty'] for loss in agent.training_losses[-500:]]
        
        axes[1,1].plot(actor_losses, label='Actor Loss', alpha=0.7)
        axes[1,1].plot(failure_penalties, label='Failure Penalty', alpha=0.7)
        axes[1,1].set_title('Training Losses (Last 500 steps)')
        axes[1,1].set_xlabel('Training Step')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].legend()
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Training Complete!")
    print(f"Final failure buffer size: {len(agent.failure_buffer)}")
    print(f"Average reward last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Your dual-buffer approach stored {len(agent.failure_buffer)} failure experiences!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-failure-buffer", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable failure buffer reward shaping and mixed sampling")
    parser.add_argument("--render-test", action=argparse.BooleanOptionalAction, default=False,
                        help="Render a short test run after training")
    args = parser.parse_args()

    # Run the experiment!
    agent, rewards, failures = train_failure_aware_sac(use_failure_buffer=args.use_failure_buffer)
    plot_results(agent, rewards, failures)
    
    if args.render_test:
        # Test the trained agent
        print("\n🎮 Testing trained agent...")
        env = gym.make('CartPole-v1', render_mode='human')
        state, _ = env.reset()
        
        for _ in range(1000):
            action, _ = agent.select_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        env.close()