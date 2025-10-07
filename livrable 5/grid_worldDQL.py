import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# --- SECTION 1: DYNAMIC GOAL ENVIRONMENT ---

class DynamicGoalGridWorldEnv:
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    def __init__(self, height=5, width=5, obstacles=[]):
        self.height = height
        self.width = width
        self.obstacle_positions = set(obstacles)
        self.agent_pos = None
        self.goal_pos = None
        self.state_space_n = 4
        self.action_space_n = 4
        self.fig, self.ax = None, None

    def reset(self):
        self.agent_pos = (0, 0)
        possible_goal_positions = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in self.obstacle_positions and (r, c) != self.agent_pos:
                    possible_goal_positions.append((r, c))
        self.goal_pos = random.choice(possible_goal_positions)
        return self._get_state()

    def step(self, action):
        d_row, d_col = self.ACTIONS[action]
        potential_new_row, potential_new_col = self.agent_pos[0] + d_row, self.agent_pos[1] + d_col
        potential_new_pos = (potential_new_row, potential_new_col)

        if (0 <= potential_new_row < self.height and
            0 <= potential_new_col < self.width and
            potential_new_pos not in self.obstacle_positions):
            self.agent_pos = potential_new_pos

        done = False
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            reward = -0.1

        return self._get_state(), reward, done, {}

    def _get_state(self):
        return np.array([
            self.agent_pos[0] / self.height, self.agent_pos[1] / self.width,
            self.goal_pos[0] / self.height, self.goal_pos[1] / self.width
        ])

    def render_live(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        for i in range(self.height + 1): self.ax.axhline(i, lw=2, color='k', zorder=1)
        for i in range(self.width + 1): self.ax.axvline(i, lw=2, color='k', zorder=1)
        for obs_pos in self.obstacle_positions: self.ax.add_patch(patches.Rectangle((obs_pos[1], obs_pos[0]), 1, 1, facecolor='dimgray', zorder=2))
        self.ax.add_patch(patches.Rectangle((self.goal_pos[1], self.goal_pos[0]), 1, 1, facecolor='gold', zorder=2))
        agent_y, agent_x = self.agent_pos
        self.ax.add_patch(patches.Circle((agent_x + 0.5, agent_y + 0.5), 0.3, facecolor='blue', zorder=3))
        self.ax.set_xlim(0, self.width); self.ax.set_ylim(0, self.height)
        self.ax.invert_yaxis(); self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_title("DQN Agent Test Drive")
        self.fig.canvas.draw(); plt.pause(0.1)


# --- SECTION 2: THE DEEP Q-NETWORK (DQN)  ---

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- SECTION 3: THE DQN AGENT (with Save/Load) ---

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.state_space_n
        self.action_dim = env.action_space_n
        self.gamma = 0.99; self.epsilon = 1.0; self.epsilon_decay = 0.995
        self.epsilon_min = 0.01; self.batch_size = 64
        self.memory = collections.deque(maxlen=10000)
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.update_target_net()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_net(state).max(1)[1].item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones).unsqueeze(1)
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        next_q_values[dones] = 0.0
        expected_q_values = rewards + (self.gamma * next_q_values)
        loss = F.mse_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target_net()
        self.epsilon = self.epsilon_min
        print(f"Model loaded from {path}")


# --- SECTION 4: PLOTTING

def plot_rewards(rewards, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Total Reward per Episode for DQN Agent')
    plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Rewards plot saved to {save_path}")
    plt.show(block=False)

def plot_loss(losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('DQN Training Loss per Learning Step')
    plt.xlabel('Learning Step'); plt.ylabel('MSE Loss'); plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    plt.show(block=False)


# --- SECTION 5: MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    RESULTS_DIR = "results.deepqlearning.dql"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_gridworld_model.pth")

    env = DynamicGoalGridWorldEnv(height=5, width=5)
    agent = DQNAgent(env)
    
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
    else:
        num_episodes = 500
        max_steps_per_episode = 100
        target_update_frequency = 10
        episode_rewards = []
        training_losses = [] 
        
        print("🚀 No saved model found. Starting training...")
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.learn()
                
                if loss is not None:
                    training_losses.append(loss)
                
                state = next_state
                total_reward += reward
                if done:
                    break
            
            episode_rewards.append(total_reward)
            if (episode + 1) % target_update_frequency == 0:
                agent.update_target_net()
                
            print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

        print("✅ Training complete.")
        agent.save(MODEL_PATH)

        plot_rewards(episode_rewards, save_path=os.path.join(RESULTS_DIR, "rewards_per_episode.png"))
        plot_loss(training_losses, save_path=os.path.join(RESULTS_DIR, "training_loss.png"))
        create_readme(RESULTS_DIR)

    # --- Evaluation ---
    num_evaluation_trials = 10
    print(f"\n🎬 Evaluating the agent for {num_evaluation_trials} random goals...")
    
    for i in range(num_evaluation_trials):
        state = env.reset()
        print(f"--- Test {i+1}/{num_evaluation_trials}: New goal at {env.goal_pos} ---")
        done = False
        for _ in range(max_steps_per_episode):
            env.render_live()
            action = agent.choose_action(state)
            state, _, done, _ = env.step(action)
            if done:
                env.render_live()
                break
    
    print("🏁 Evaluation finished.")
    plt.ioff()
    plt.show()