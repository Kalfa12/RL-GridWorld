import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collections
import time
from IPython.display import display, clear_output
import os  
import imageio

# --- SECTION 1: CLASSE DE L'ENVIRONNEMENT ET DE L'AGENT ---
class GridWorldEnv:
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    def __init__(self, height, width, start_pos=(0, 0), goals={(3, 3): 1.0}, obstacles=[]):
        self.height = height
        self.width = width
        self.observation_space_n = height * width
        self.action_space_n = 4
        self.start_pos = start_pos
        self.goals = goals
        self.obstacle_positions = set(obstacles)
        self.goal_positions = set(self.goals.keys())
        if self.start_pos in self.obstacle_positions or self.start_pos in self.goal_positions:
            raise ValueError("Start position cannot be an obstacle or a goal.")
        if not self.goal_positions.isdisjoint(self.obstacle_positions):
            raise ValueError("Goals and obstacles cannot overlap.")
        self.agent_pos = None
        self.fig, self.ax = None, None

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_state_from_pos(self.agent_pos)

    def step(self, action):
        current_row, current_col = self.agent_pos
        d_row, d_col = self.ACTIONS[action]
        potential_new_row, potential_new_col = current_row + d_row, current_col + d_col
        potential_new_pos = (potential_new_row, potential_new_col)
        
        if (0 <= potential_new_row < self.height and
            0 <= potential_new_col < self.width and
            potential_new_pos not in self.obstacle_positions):
            self.agent_pos = potential_new_pos
        
        reward = -0.01
        done = False
        if self.agent_pos in self.goal_positions:
            reward = self.goals[self.agent_pos]
            done = True
        
        next_state = self._get_state_from_pos(self.agent_pos)
        return next_state, reward, done, {}

    def _get_state_from_pos(self, pos):
        return pos[0] * self.width + pos[1]

    def _get_pos_from_state(self, state):
        return (state // self.width, state % self.width)
    
    def render_live(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        for i in range(self.height + 1): self.ax.axhline(i, lw=2, color='k', zorder=1)
        for i in range(self.width + 1): self.ax.axvline(i, lw=2, color='k', zorder=1)
        for obs_pos in self.obstacle_positions: self.ax.add_patch(patches.Rectangle((obs_pos[1], obs_pos[0]), 1, 1, facecolor='dimgray', zorder=2))
        for goal_pos in self.goal_positions: self.ax.add_patch(patches.Rectangle((goal_pos[1], goal_pos[0]), 1, 1, facecolor='gold', zorder=2))
        agent_y, agent_x = self.agent_pos
        self.ax.add_patch(patches.Circle((agent_x + 0.5, agent_y + 0.5), 0.3, facecolor='blue', zorder=3))
        self.ax.set_xlim(0, self.width); self.ax.set_ylim(0, self.height)
        self.ax.invert_yaxis(); self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_title("Agent's Test Drive")
        self.fig.canvas.draw(); plt.pause(0.1)

    def render_policy_and_values(self, policy=None, values=None, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(self.height + 1): ax.axhline(i, lw=2, color='k')
        for i in range(self.width + 1): ax.axvline(i, lw=2, color='k')
        for goal_pos in self.goal_positions: ax.add_patch(patches.Rectangle((goal_pos[1], goal_pos[0]), 1, 1, facecolor='gold'))
        if hasattr(self, 'obstacle_positions'):
            for obs_pos in self.obstacle_positions: ax.add_patch(patches.Rectangle((obs_pos[1], obs_pos[0]), 1, 1, facecolor='dimgray'))
        for state in range(self.observation_space_n):
            r, c = self._get_pos_from_state(state)
            if (r, c) in self.goal_positions or (r, c) in self.obstacle_positions: continue
            if policy is not None and state in policy:
                ax.text(c + 0.5, r + 0.7, self.ACTION_NAMES[policy[state]], ha='center', va='center', fontsize=16, color='black')
        ax.set_xlim(0, self.width); ax.set_ylim(0, self.height)
        ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Final Policy")
        if save_path:
            plt.savefig(save_path)
            print(f"Policy plot saved to {save_path}")
        plt.ioff(); plt.show()


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05):
        self.env = env; self.alpha = alpha; self.gamma = gamma
        self.epsilon = epsilon; self.epsilon_decay = epsilon_decay; self.epsilon_min = epsilon_min
        self.q_table = collections.defaultdict(lambda: np.zeros(env.action_space_n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon: return np.random.randint(self.env.action_space_n)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state]) if not done else 0.0
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
#
# --- SECTION 2: FONCTIONS D'AIDE POUR L'ENTRAÎNEMENT ET LA VISUALISATION ---
#

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_model_free_agent(agent, env, num_episodes=2000, max_steps_per_episode=200):
    print(f"🚀 Training on {env.height}x{env.width} grid for {num_episodes} episodes...")
    episode_rewards = []; epsilon_history = []
    for episode in range(num_episodes):
        state = env.reset(); done = False; total_reward = 0; step_count = 0
        while not done and step_count < max_steps_per_episode:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.learn(state, action, reward, next_state, done)
            state = next_state; step_count += 1
        agent.decay_epsilon()
        episode_rewards.append(total_reward); epsilon_history.append(agent.epsilon)
    print("✅ Training complete.")
    return episode_rewards, epsilon_history

def run_evaluation_episode(agent, env, gif_path=None):
    print("\n🎬 Animation de l'agent entraîné...")
    agent.epsilon = 0
    state = env.reset()
    done = False
    frames = []

    while not done:
        env.render_live()
        if gif_path:
            # Capture frame
            fig = plt.gcf()
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
        
        action = agent.choose_action(state)
        state, _, done, _ = env.step(action)
    
    env.render_live() # Render the final state
    if gif_path:
        # Capture the final frame
        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        # Save GIF
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"Animation saved to {gif_path}")

    print("🏁 Épisode terminé.")

def plot_rewards(rewards, agent_name, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f'Rewards per Episode for {agent_name}')
    plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Rewards plot saved to {save_path}")
    plt.show(block=False)

def plot_epsilon(epsilon_history, grid_size_str, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_history)
    plt.title(f'Decay de Epsilon ({grid_size_str})')
    plt.xlabel('Épisode'); plt.ylabel('Valeur de Epsilon'); plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Epsilon plot saved to {save_path}")
    plt.show()

def plot_comparison(results, smoothing_window=50, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.set_title('Comparaison du retour cumulé (Brut)'); ax1.set_ylabel('Retour cumulé')
    for label, (rewards, _) in results.items():
        ax1.plot(rewards, label=label)
    ax1.grid(True); ax1.legend()
    ax2.set_title(f'Comparaison du retour cumulé (Lissé, fenêtre={smoothing_window})')
    ax2.set_xlabel('Épisode'); ax2.set_ylabel('Retour cumulé (moyenne mobile)')
    for label, (rewards, _) in results.items():
        smoothed_rewards = pd.Series(rewards).rolling(window=smoothing_window, min_periods=1).mean()
        ax2.plot(smoothed_rewards, label=label)
    ax2.grid(True); ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    plt.show(block=False)


#
# --- SECTION 3: EXÉCUTION PRINCIPALE AVEC MENU ---
#

if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print(" MENU PRINCIPAL - GRID WORLD ")
        print("="*40)
        print("1. Lancer une seule simulation interactive (et sauvegarder les résultats)")
        print("2. Lancer des expériences comparatives (et sauvegarder les résultats)")
        print("3. Quitter")
        choice = input("Votre choix : ")

        if choice == '1':
            # --- Configuration pour la simulation unique ---
            env = GridWorldEnv(
                height=5, width=10,
                goals={(4, 9): 10.0, (0, 9): 1.0},
                obstacles=[(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
            )
            agent = QLearningAgent(env)
            rewards, epsilons = run_model_free_agent(agent, env, num_episodes=2000, max_steps_per_episode=150)
            
            final_policy = {s: np.argmax(q_vals) for s, q_vals in agent.q_table.items() if s in agent.q_table}
            env.render_policy_and_values(final_policy, save_path=os.path.join(RESULTS_DIR, 'single_run_policy.png'))
            run_evaluation_episode(agent, env, gif_path=os.path.join(RESULTS_DIR, 'trained_agent.gif'))
            
            plot_rewards(rewards, "QLearningAgent", save_path=os.path.join(RESULTS_DIR, 'single_run_rewards.png'))
            plot_epsilon(epsilons, "5x10 Grid", save_path=os.path.join(RESULTS_DIR, 'single_run_epsilon.png'))
            plt.show()

        elif choice == '2':
            # --- Configuration pour les expériences comparatives ---
            grid_sizes_to_test = [4, 6, 8]
            NUM_EPISODES = 2000
            MAX_STEPS = 150
            SMOOTHING_WINDOW = 50
            results = {}
            last_epsilon_history = None

            for size in grid_sizes_to_test:
                print("-" * 30)
                env = GridWorldEnv(height=size, width=size, goals={(size - 1, size - 1): 10.0})
                agent = QLearningAgent(env)
                rewards, epsilons = run_model_free_agent(agent, env, num_episodes=NUM_EPISODES, max_steps_per_episode=MAX_STEPS)
                results[f'Grille {size}x{size}'] = (rewards, epsilons)
                last_epsilon_history = epsilons
            
            print("\n" + "="*30 + "\nExpériences terminées. Affichage des graphiques...\n" + "="*30)
            
            plot_comparison(results, smoothing_window=SMOOTHING_WINDOW, save_path=os.path.join(RESULTS_DIR, 'comparison_rewards.png'))
            if last_epsilon_history:
                last_size_str = f"Grille {grid_sizes_to_test[-1]}x{grid_sizes_to_test[-1]}"
                plot_epsilon(last_epsilon_history, last_size_str, save_path=os.path.join(RESULTS_DIR, 'last_run_epsilon_decay.png'))
            plt.show()

        elif choice == '3':
            print("Au revoir !")
            break
        else:
            print("Choix invalide, veuillez réessayer.")