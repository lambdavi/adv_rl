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
        # Convert action to appropriate format for storage
        if isinstance(action, (int, np.integer)):
            # Discrete action - store as integer
            action = int(action)
        elif isinstance(action, np.ndarray):
            # Continuous action - store as array
            action = action.copy()
        else:
            # Fallback - convert to appropriate type
            action = np.array(action)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors, handling different action types
        states_tensor = torch.FloatTensor(np.array(states))
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        dones_tensor = torch.FloatTensor(np.array(dones))
        
        # Handle actions - could be mixed types
        if all(isinstance(a, (int, np.integer)) for a in actions):
            # All discrete actions
            actions_tensor = torch.LongTensor(actions)
        else:
            # Mixed or continuous actions
            actions_tensor = torch.FloatTensor(actions)
        
        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
    
    def __len__(self):
        return len(self.buffer)

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
    
    def compute_penalty(self, state):
        if not self.clusters:
            return 0.0
        
        state = np.array(state)
        min_distance = float('inf')
        
        for centroid, states, weight in self.clusters:
            distance = np.linalg.norm(state - centroid)
            # Weight by cluster size (bigger clusters = more dangerous)
            weighted_distance = distance / (weight ** 0.5)
            min_distance = min(min_distance, weighted_distance)
        
        # Exponential penalty based on distance
        return np.exp(-min_distance)
    
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space_type, hidden_dim=256):
        super(Actor, self).__init__()
        self.action_space_type = action_space_type
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if action_space_type == 'discrete':
            # For discrete actions, output logits for each action
            self.fc3 = nn.Linear(hidden_dim, action_dim)
        else:
            # For continuous actions, output mean and log_std for each action dimension
            self.fc3_mean = nn.Linear(hidden_dim, action_dim)
            self.fc3_logstd = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.action_space_type == 'discrete':
            # Output logits for discrete actions
            logits = self.fc3(x)
            return logits
        else:
            # Output mean and log_std for continuous actions
            mean = self.fc3_mean(x)
            log_std = self.fc3_logstd(x)
            log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
            return mean, log_std
    
    def sample_action(self, state):
        if self.action_space_type == 'discrete':
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, probs
        else:
            mean, log_std = self.forward(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # Use rsample for reparameterization
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, (mean, std)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, action_space_type, hidden_dim=256):
        super(Critic, self).__init__()
        self.action_space_type = action_space_type
        
        if action_space_type == 'discrete':
            # For discrete actions, input is just state
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)  # Q-value for each action
        else:
            # For continuous actions, input is state + action
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action=None):
        if self.action_space_type == 'discrete':
            # For discrete actions, return Q-values for all actions
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
            return q_values
        else:
            # For continuous actions, concatenate state and action
            x = torch.cat([state, action], dim=-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value = self.fc3(x)
            return q_value

class FailureAwareSAC:
    def __init__(self, state_dim, action_dim, action_space_type, lr=3e-4, gamma=0.99, tau=0.005, use_failure_buffer=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.use_failure_buffer = use_failure_buffer
        self.action_space_type = action_space_type
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim, action_space_type).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, action_space_type).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, action_space_type).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim, action_space_type).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, action_space_type).to(self.device)
        
        # Copy target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Temperature parameter for entropy regularization
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Target entropy for temperature tuning
        if action_space_type == 'discrete':
            self.target_entropy = -np.log(1.0 / action_dim)  # For discrete uniform policy
        else:
            self.target_entropy = -action_dim  # For continuous actions
        
        # Buffers
        self.replay_buffer = ReplayBuffer(100000)
        if use_failure_buffer:
            self.failure_cluster_buffer = FailureClusterBuffer()
        
        # Training tracking
        self.training_losses = []
        self.failure_penalties = []
        
        # Failure buffer parameters
        self.failure_penalty_weight = 0.5  # beta for reward shaping
        self.failure_distance_scale = 2.0   # Scale for distance computation
        self.mixed_failure_fraction = 0.3   # fraction of minibatch from failure buffer
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            if self.action_space_type == 'discrete':
                with torch.no_grad():
                    logits = self.actor(state)
                    probs = F.softmax(logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)
            else:
                with torch.no_grad():
                    mean, _ = self.actor(state)
                    action = torch.tanh(mean)  # Clamp to [-1, 1]
        else:
            action, _, _ = self.actor.sample_action(state)
            if self.action_space_type == 'continuous':
                action = torch.tanh(action)  # Clamp to [-1, 1]
        
        # Return appropriate format based on action space type
        if self.action_space_type == 'discrete':
            return action.cpu().numpy().item()  # Return scalar for discrete actions
        else:
            return action.cpu().numpy().flatten()  # Return array for continuous actions
    
    def compute_failure_penalty(self, states):
        if not self.use_failure_buffer or not self.failure_cluster_buffer.clusters:
            return torch.zeros(states.shape[0], device=self.device)
        
        penalties = []
        for state in states:
            penalty = self.failure_cluster_buffer.compute_penalty(state.cpu().numpy())
            penalties.append(penalty)
        
        return torch.FloatTensor(penalties).to(self.device) * self.failure_penalty_weight
    
    def sample_mixed_minibatch(self, batch_size):
        if not self.use_failure_buffer or not self.failure_cluster_buffer.clusters:
            return self.replay_buffer.sample(batch_size)
        
        # Sample from regular buffer
        regular_batch_size = int(batch_size * (1 - self.mixed_failure_fraction))
        regular_batch = self.replay_buffer.sample(regular_batch_size)
        
        # Sample from failure clusters
        failure_batch_size = batch_size - regular_batch_size
        danger_states = self.failure_cluster_buffer.sample_danger_states(failure_batch_size)
        
        if danger_states:
            # Create synthetic experiences from danger states
            failure_batch = []
            for state in danger_states:
                # Random action, negative reward, same state (terminal)
                if self.action_space_type == 'discrete':
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = np.random.uniform(-1, 1, self.action_dim)
                
                failure_batch.append((state, action, -10.0, state, 1.0))
            
            # Combine batches
            all_states = list(regular_batch[0]) + [exp[0] for exp in failure_batch]
            all_actions = list(regular_batch[1]) + [exp[1] for exp in failure_batch]
            all_rewards = list(regular_batch[2]) + [exp[2] for exp in failure_batch]
            all_next_states = list(regular_batch[3]) + [exp[3] for exp in failure_batch]
            all_dones = list(regular_batch[4]) + [exp[4] for exp in failure_batch]
            
            return (torch.FloatTensor(all_states), torch.FloatTensor(all_actions),
                    torch.FloatTensor(all_rewards), torch.FloatTensor(all_next_states),
                    torch.FloatTensor(all_dones))
        else:
            return regular_batch
    
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample mixed minibatch
        states, actions, rewards, next_states, dones = self.sample_mixed_minibatch(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Reward shaping via failure proximity (detach to avoid gradients through reward)
        with torch.no_grad():
            penalties = self.compute_failure_penalty(states)
            shaped_rewards = rewards - penalties
            penalty_mean = penalties.mean().item() if penalties.numel() > 0 else 0.0
        
        # Update critics
        with torch.no_grad():
            if self.action_space_type == 'discrete':
                # For discrete actions, compute target Q using next state action probabilities
                next_action, next_log_prob, next_probs = self.actor.sample_action(next_states)
                next_q1 = self.target_critic1(next_states)
                next_q2 = self.target_critic2(next_states)
                
                # Expected Q-value over next action distribution
                next_q1_exp = (next_probs * next_q1).sum(dim=-1)
                next_q2_exp = (next_probs * next_q2).sum(dim=-1)
                next_q = torch.min(next_q1_exp, next_q2_exp)
                
                # Add entropy term
                alpha = self.log_alpha.exp()
                next_q = next_q - alpha * (-(next_probs * torch.log(next_probs + 1e-8)).sum(dim=-1))
            else:
                # For continuous actions
                next_action, next_log_prob, _ = self.actor.sample_action(next_states)
                next_q1 = self.target_critic1(next_states, next_action)
                next_q2 = self.target_critic2(next_states, next_action)
                next_q = torch.min(next_q1, next_q2)
                
                # Add entropy term
                alpha = self.log_alpha.exp()
                next_q = next_q - alpha * next_log_prob
            
            target_q = shaped_rewards + self.gamma * (1 - dones) * next_q
        
        # Current Q-values
        if self.action_space_type == 'discrete':
            current_q1 = self.critic1(states)
            current_q2 = self.critic2(states)
            # Get Q-values for taken actions
            current_q1 = current_q1.gather(1, actions.long().unsqueeze(1)).squeeze(1)
            current_q2 = current_q2.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        else:
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        if self.action_space_type == 'discrete':
            # For discrete actions
            action, log_prob, probs = self.actor.sample_action(states)
            q1 = self.critic1(states)
            q2 = self.critic2(states)
            q = torch.min(q1, q2)
            
            # Expected Q-value over action distribution
            q_exp = (probs * q).sum(dim=-1)
            
            # Actor loss with entropy
            alpha = self.log_alpha.exp()
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            actor_loss = (alpha * log_prob - q_exp).mean()
        else:
            # For continuous actions
            action, log_prob, _ = self.actor.sample_action(states)
            q1 = self.critic1(states, action)
            q2 = self.critic2(states, action)
            q = torch.min(q1, q2)
            
            # Actor loss with entropy
            alpha = self.log_alpha.exp()
            actor_loss = (alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        if self.action_space_type == 'discrete':
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        else:
            entropy = -log_prob
        
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log training info
        self.training_losses.append({
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'failure_penalty': penalty_mean
        })
        
        self.failure_penalties.append(penalty_mean)

def train_failure_aware_sac(use_failure_buffer=True, episodes=1000, max_steps=500):
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    
    # Dynamically determine action space type and dimensions
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_space_type = 'discrete'
        action_dim = env.action_space.n
    else:
        action_space_type = 'continuous'
        action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env.spec.id}")
    print(f"State dimension: {state_dim}")
    print(f"Action space type: {action_space_type}")
    print(f"Action dimension: {action_dim}")
    
    agent = FailureAwareSAC(state_dim, action_dim, action_space_type, use_failure_buffer=use_failure_buffer)
    
    episode_rewards = []
    failure_counts = []
    
    print("🚀 Training Failure-Aware SAC on LunarLander!")
    mode = "ENABLED" if use_failure_buffer else "DISABLED"
    print(f"💡 Failure buffer is {mode} (danger zone clustering + mixed sampling)")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_failures = 0
        trajectory = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)
            trajectory.append((state, action, reward, next_state, done))
            
            # Check for failure (negative reward or crash)
            if reward < -50 or done and reward < 0:
                episode_failures += 1
                if agent.use_failure_buffer:
                    # Add final states from trajectory to failure buffer
                    for traj_state, _, _, _, _ in trajectory[-5:]:  # Last 5 states
                        agent.failure_cluster_buffer.add_failure_state(traj_state)
            
            episode_reward += reward
            state = next_state
            
            # Update agent
            agent.update()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        failure_counts.append(episode_failures)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_failures = np.mean(failure_counts[-100:])
            failure_buffer_size = len(agent.failure_cluster_buffer.clusters) if agent.use_failure_buffer else 0
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Failures = {avg_failures:.2f}, Failure Clusters = {failure_buffer_size}")
    
    env.close()
    return agent, episode_rewards, failure_counts

def plot_results(agent, rewards, failures):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0,0].plot(rewards, alpha=0.6)
    axes[0,0].set_title('Episode Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].grid(True)
    
    # Moving average of rewards
    window = 100
    if len(rewards) >= window:
        moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        axes[0,1].plot(moving_avg, label=f'{window}-episode moving average', linewidth=2)
    axes[0,1].set_title('Moving Average Reward')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Average Reward')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Failure counts
    axes[1,0].plot(failures, alpha=0.6, color='red')
    axes[1,0].set_title('Episode Failures')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Number of Failures')
    axes[1,0].grid(True)
    
    # Training losses
    if agent.training_losses:
        last_losses = agent.training_losses[-500:]
        actor_losses = [loss['actor_loss'] for loss in last_losses]
        # Prefer logged failure_penalty, fallback to agent.failure_penalties
        failure_penalties = []
        for loss in last_losses:
            penalty = loss.get('failure_penalty')
            if penalty is not None:
                failure_penalties.append(penalty)
        
        if not failure_penalties and agent.failure_penalties:
            failure_penalties = agent.failure_penalties[-len(actor_losses):] if agent.failure_penalties else []
        
        axes[1,1].plot(actor_losses, label='Actor Loss', alpha=0.7)
        if failure_penalties:
            axes[1,1].plot(failure_penalties, label='Failure Penalty', alpha=0.7)
        axes[1,1].set_title('Training Losses (Last 500 steps)')
        axes[1,1].set_xlabel('Training Step')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].legend()
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Failure-Aware SAC on LunarLander')
    parser.add_argument("--use-failure-buffer", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable failure buffer (danger zone clustering)")
    parser.add_argument("--render-test", action=argparse.BooleanOptionalAction, default=False,
                        help="Render a short test run after training")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train for")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum number of steps per episode")
    args = parser.parse_args()
    
    # Train the agent
    agent, rewards, failures = train_failure_aware_sac(
        use_failure_buffer=args.use_failure_buffer,
        episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    # Plot results
    plot_results(agent, rewards, failures)
    
    # Optional: render a test run
    if args.render_test:
        print("\n🎬 Rendering test run...")
        env = gym.make('LunarLander-v3', render_mode='human')
        state, _ = env.reset()
        
        for _ in range(1000):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            if done:
                state, _ = env.reset()
        
        env.close()