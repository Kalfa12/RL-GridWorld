import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import collections
from IPython.display import display, clear_output

# actions 0: Up, 1: Down, 2: Left, 3: Right
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: '↑', 1: '↓', 2: '←', 3: '→'}

####################################################################
# SECTION 1: THE ENVIRONMENT CLASS
####################################################################
class GridWorldEnv:
    def __init__(self, size=5):
        self.size = size
        self.observation_space_n = size * size
        self.action_space_n = 4
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.agent_pos = None
        
        # Visualization objects
        self.fig = None
        self.ax = None

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_state_from_pos(self.agent_pos)

    def step(self, action):
        current_row, current_col = self.agent_pos
        d_row, d_col = ACTIONS[action]
        new_row = current_row + d_row
        new_col = current_col + d_col
        
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = (new_row, new_col)
        
        done = (self.agent_pos == self.goal_pos)
        reward = 1.0 if done else -0.01
            
        next_state = self._get_state_from_pos(self.agent_pos)
        return next_state, reward, done, {}

    def get_model(self):
        model = collections.defaultdict(dict)
        for r in range(self.size):
            for c in range(self.size):
                state = self._get_state_from_pos((r, c))
                for action in range(self.action_space_n):
                    self.agent_pos = (r, c)
                    next_state, reward, done, _ = self.step(action)
                    model[state][action] = [(1.0, next_state, reward, done)]
        return model
        
    def _get_state_from_pos(self, pos):
        return pos[0] * self.size + pos[1]

    def _get_pos_from_state(self, state):
        return (state // self.size, state % self.size)

    def render_live(self):
        """Renders the agent moving in the grid."""
        if self.fig is None:
            # Use interactive mode
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        
        for i in range(self.size + 1):
            self.ax.axhline(i, lw=2, color='k', zorder=1)
            self.ax.axvline(i, lw=2, color='k', zorder=1)

        goal_y, goal_x = self.goal_pos
        self.ax.add_patch(patches.Rectangle((goal_x, goal_y), 1, 1, facecolor='gold', zorder=2))
        
        agent_y, agent_x = self.agent_pos
        self.ax.add_patch(patches.Circle((agent_x + 0.5, agent_y + 0.5), 0.3, facecolor='blue', zorder=3))

        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Agent's Test Drive")
        
        # Redraw canvas and pause
        self.fig.canvas.draw()
        plt.pause(0.2)
        
    def render_policy_and_values(self, policy=None, values=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(self.size + 1):
            ax.axhline(i, lw=2, color='k')
            ax.axvline(i, lw=2, color='k')

        ax.add_patch(patches.Rectangle(self.goal_pos[::-1], 1, 1, facecolor='gold'))
        
        for state in range(self.observation_space_n):
            r, c = self._get_pos_from_state(state)
            if (r, c) == self.goal_pos: continue
            
            if values is not None:
                ax.text(c + 0.5, r + 0.2, f'{values[state]:.2f}', ha='center', va='center', fontsize=8,color='black')
            if policy is not None:
                ax.text(c + 0.5, r + 0.7, ACTION_NAMES[policy[state]], ha='center', va='center', fontsize=16, color='black')

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Final Policy")
        plt.show(block=False)

####################################################################
# SECTION 2: THE AGENT CLASSES (UNCHANGED)
####################################################################
class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros(env.observation_space_n)
        self.policy = np.zeros(env.observation_space_n, dtype=int)
        self.model = env.get_model()

    def learn(self):
        while True:
            delta = 0
            for s in range(self.env.observation_space_n):
                v = self.values[s]
                action_values = [sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[s][a]]) for a in range(self.env.action_space_n)]
                self.values[s] = max(action_values)
                delta = max(delta, abs(v - self.values[s]))
            if delta < self.theta:
                break
        
        for s in range(self.env.observation_space_n):
            action_values = [sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[s][a]]) for a in range(self.env.action_space_n)]
            self.policy[s] = np.argmax(action_values)
            
    def choose_action(self, state):
        return self.policy[state]

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros(env.observation_space_n)
        self.policy = np.random.randint(0, env.action_space_n, env.observation_space_n)
        self.model = env.get_model()

    def learn(self):
        while True:
            while True:
                delta = 0
                for s in range(self.env.observation_space_n):
                    v = self.values[s]
                    action = self.policy[s]
                    self.values[s] = sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[s][action]])
                    delta = max(delta, abs(v - self.values[s]))
                if delta < self.theta:
                    break
            
            policy_stable = True
            for s in range(self.env.observation_space_n):
                old_action = self.policy[s]
                action_values = [sum([p * (r + self.gamma * self.values[s_]) for p, s_, r, _ in self.model[s][a]]) for a in range(self.env.action_space_n)]
                self.policy[s] = np.argmax(action_values)
                if old_action != self.policy[s]:
                    policy_stable = False
            
            if policy_stable:
                break

    def choose_action(self, state):
        return self.policy[state]

class MonteCarloAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = collections.defaultdict(lambda: np.zeros(env.action_space_n))
        self.returns = collections.defaultdict(list)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space_n)
        return np.argmax(self.q_table[state])

    def learn(self, episode):
        G = 0
        visited_sa_pairs = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited_sa_pairs:
                self.returns[(state, action)].append(G)
                self.q_table[state][action] = np.mean(self.returns[(state, action)])
                visited_sa_pairs.add((state, action))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = collections.defaultdict(lambda: np.zeros(env.action_space_n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space_n)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

####################################################################
# SECTION 3: TRAINING & VISUALIZATION
####################################################################

def plot_rewards(rewards, agent_name):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f'Rewards per Episode for {agent_name}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show(block=False)

def run_evaluation_episode(agent, env):
    """Animates one episode of a trained agent."""
    print("\n🎬 Now watching the trained agent...")
    # For model-free agents, we want to see the greedy policy in action
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0 # Turn off exploration
        
    state = env.reset()
    done = False
    
    while not done:
        env.render_live()
        action = agent.choose_action(state)
        state, _, done, _ = env.step(action)
    
    env.render_live() # Render the final state
    print("🏁 Episode finished.")
    plt.ioff() # Turn off interactive mode
    plt.show() # Show all plots and wait for user to close

def run_model_based_agent(agent, env):
    print(f"🧠 Solving with {agent.__class__.__name__}...")
    agent.learn()
    print("✅ Policy and Value Function Learned.")
    env.render_policy_and_values(agent.policy, agent.values)
    run_evaluation_episode(agent, env)

def run_model_free_agent(agent, env, num_episodes=5000):
    print(f"🚀 Training {agent.__class__.__name__} for {num_episodes} episodes...")
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_history = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if isinstance(agent, QLearningAgent):
                agent.learn(state, action, reward, next_state)
            elif isinstance(agent, MonteCarloAgent):
                episode_history.append((state, action, reward))
            
            state = next_state
        
        if isinstance(agent, MonteCarloAgent):
            agent.learn(episode_history)
        if isinstance(agent, QLearningAgent):
            agent.decay_epsilon()
            
        episode_rewards.append(total_reward)
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")

    print("✅ Training complete.")
    plot_rewards(episode_rewards, agent.__class__.__name__)
    
    final_policy = {s: np.argmax(q_vals) for s, q_vals in agent.q_table.items()}
    full_policy = np.zeros(env.observation_space_n, dtype=int)
    for s in range(env.observation_space_n):
        if s in final_policy: full_policy[s] = final_policy[s]

    env.render_policy_and_values(full_policy, None)
    run_evaluation_episode(agent, env)

####################################################################
# SECTION 4: MAIN EXECUTION (UPDATED)
####################################################################
if __name__ == "__main__":
    env = GridWorldEnv(size=5)
    
    while True:
        print("\n--- RL Agent Selection Menu ---")
        print("1: Value Iteration")
        print("2: Policy Iteration")
        print("3: Monte Carlo")
        print("4: Q-Learning")
        choice = input("Enter the number of the agent to run: ")
        
        if choice in ['1', '2', '3', '4']:
            break
        else:
            print("Invalid input. Please enter a number from 1 to 4.")

    if choice == '1':
        agent = ValueIterationAgent(env)
        run_model_based_agent(agent, env)
    elif choice == '2':
        agent = PolicyIterationAgent(env)
        run_model_based_agent(agent, env)
    elif choice == '3':
        agent = MonteCarloAgent(env)
        run_model_free_agent(agent, env, num_episodes=5000)
    elif choice == '4':
        agent = QLearningAgent(env)
        run_model_free_agent(agent, env, num_episodes=5000)