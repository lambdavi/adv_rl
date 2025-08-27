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
        # For CartPole: output probability of going right
        action_prob = torch.sigmoid(self.fc3(x))
        return action_prob

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
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        
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
        
        # Replay buffers - YOUR NOVEL DUAL BUFFER APPROACH!
        self.experience_buffer = ReplayBuffer(100000)
        self.failure_buffer = ReplayBuffer(10000)  # Separate buffer for failures
        
        # Adversarial parameters
        self.failure_penalty_weight = 0.5  # How much to penalize approaching failures
        self.failure_distance_scale = 2.0   # Scale for distance computation
        
        # Logging
        self.failure_penalties = []
        self.training_losses = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_prob = self.actor(state)
        
        # For CartPole: sample action based on probability
        if random.random() < action_prob.cpu().data.numpy()[0][0]:
            action = 1  # Right
        else:
            action = 0  # Left
            
        return action, action_prob.cpu().data.numpy()[0]
    
    def store_experience(self, state, action, reward, next_state, done):
        # Store in regular experience buffer
        action_continuous = np.array([action], dtype=np.float32)
        self.experience_buffer.push(state, action_continuous, reward, next_state, done)
        
        # YOUR CORE INNOVATION: Store failures separately!
        if done and reward <= 0:  # CartPole failure
            #print(f"🚨 Failure stored! State: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}, {state[3]:.3f}]")
            self.failure_buffer.push(state, action_continuous, reward, next_state, done)
    
    def compute_failure_penalty(self, current_states):
        """YOUR KEY INNOVATION: Adversarial penalty for approaching failure states"""
        if len(self.failure_buffer) < 10:
            return torch.tensor(0.0, device=self.device)
        
        # Sample failure states
        failure_states = self.failure_buffer.sample_states(min(32, len(self.failure_buffer)))
        failure_states = torch.FloatTensor(failure_states).to(self.device)
        
        # Compute distances between current states and failure states
        current_expanded = current_states.unsqueeze(1)  # [batch, 1, state_dim]
        failure_expanded = failure_states.unsqueeze(0)   # [1, failure_batch, state_dim]
        
        # L2 distance in state space
        distances = torch.norm(current_expanded - failure_expanded, dim=2, p=2)  # [batch, failure_batch]
        min_distances = torch.min(distances, dim=1)[0]  # Closest failure state for each current state
        
        # Exponential penalty - stronger as you get closer to failure states
        penalties = self.failure_penalty_weight * torch.exp(-min_distances / self.failure_distance_scale)
        avg_penalty = penalties.mean()
        
        # Log for analysis
        self.failure_penalties.append(avg_penalty.item())
        
        return avg_penalty
    
    def update(self, batch_size=64):
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample from experience buffer
        states, actions, rewards, next_states, dones = self.experience_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # Critic update (standard SAC)
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_actions = torch.where(torch.rand_like(next_action_probs) < next_action_probs, 
                                     torch.ones_like(next_action_probs), 
                                     torch.zeros_like(next_action_probs))
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * target_q * (~dones)
        
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
        
        # Actor update with YOUR FAILURE-AWARE ADVERSARIAL TERM!
        action_probs = self.actor(states)
        sampled_actions = torch.where(torch.rand_like(action_probs) < action_probs,
                                    torch.ones_like(action_probs),
                                    torch.zeros_like(action_probs))
        
        # Standard actor loss
        actor_q1 = self.critic1(states, sampled_actions)
        actor_q2 = self.critic2(states, sampled_actions)
        actor_q = torch.min(actor_q1, actor_q2)
        actor_loss = -actor_q.mean()
        
        # YOUR INNOVATION: Add adversarial failure penalty!
        failure_penalty = self.compute_failure_penalty(states)
        total_actor_loss = actor_loss + failure_penalty
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log training info
        self.training_losses.append({
            'actor_loss': actor_loss.item(),
            'failure_penalty': failure_penalty.item(),
            'total_actor_loss': total_actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item()
        })

def train_failure_aware_sac():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # Continuous representation of discrete action
    
    agent = FailureAwareSAC(state_dim, action_dim)
    
    episodes = 1000
    max_steps = 500
    episode_rewards = []
    failure_counts = []
    
    print("🚀 Training Failure-Aware SAC on CartPole!")
    print("💡 Watch for failure states being stored and penalties being applied!")
    
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
    # Run the experiment!
    agent, rewards, failures = train_failure_aware_sac()
    plot_results(agent, rewards, failures)
    
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