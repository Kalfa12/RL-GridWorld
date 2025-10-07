import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from IPython.display import display, clear_output

# actions 0: Up, 1: Down, 2: Left, 3: Right
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

class GridWorldEnv:

    def __init__(self, size=5):
        self.size = size
        self.observation_space_n = size * size
        self.action_space_n = 4
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.agent_pos = None
        self.fig = None
        self.ax_grid = None     
        self.ax_reward = None

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_state_from_pos(self.agent_pos)

    def step(self, action):
        """Executes one time step"""
        current_row, current_col = self.agent_pos
        d_row, d_col = ACTIONS[action]
        new_row = current_row + d_row
        new_col = current_col + d_col
        
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = (new_row, new_col)
        
        done = False
        
        if self.agent_pos == self.goal_pos:
            reward = 1.0
        else:
            reward = -0.01
            
        next_state = self._get_state_from_pos(self.agent_pos)
        info = {}
        
        return next_state, reward, done, info

    def render(self, cumulative_rewards):
        """Renders the environment and the reward plot."""
        if self.fig is None:
            self.fig, (self.ax_grid, self.ax_reward) = plt.subplots(1, 2, figsize=(12, 6))

        #Plot 1: The Grid World
        self.ax_grid.clear()

        for i in range(self.size + 1):
            self.ax_grid.axhline(i, lw=2, color='k', zorder=1)
            self.ax_grid.axvline(i, lw=2, color='k', zorder=1)

        goal_y, goal_x = self.goal_pos
        self.ax_grid.add_patch(patches.Rectangle((goal_x, goal_y), 1, 1, facecolor='green', alpha=0.7, zorder=2))
        
        agent_y, agent_x = self.agent_pos
        agent_center_x = agent_x + 0.5
        agent_center_y = agent_y + 0.5
        self.ax_grid.add_patch(patches.Circle((agent_center_x, agent_center_y), 0.3, facecolor='blue', zorder=3))

        self.ax_grid.set_xlim(0, self.size)
        self.ax_grid.set_ylim(0, self.size)
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.ax_grid.set_title("5x5 Grid World")

        #Plot 2: Cumulative Reward
        self.ax_reward.clear()
        self.ax_reward.plot(cumulative_rewards, color='purple')
        self.ax_reward.set_title("Cumulative Reward Over Time")
        self.ax_reward.set_xlabel("Step")
        self.ax_reward.set_ylabel("Total Reward")
        self.ax_reward.grid(True, linestyle='--', alpha=0.6)
        
        #both plots
        self.fig.canvas.draw()
        display(self.fig)
        clear_output(wait=True)
        time.sleep(0.1)

    def _get_state_from_pos(self, pos):
        """Converts a (row, col) to a state."""
        return pos[0] * self.size + pos[1]

#        Main

if __name__ == "__main__":
    env = GridWorldEnv(size=5)
    
    state = env.reset()
    total_reward = 0
    cumulative_rewards = []
    
    MAX_STEPS = 20
    
    print(f"starting random walk for {MAX_STEPS} steps...")
    
    # Initial render with an empty rewards list
    env.render(cumulative_rewards)
    
    for step_count in range(1, MAX_STEPS + 1):
        action = np.random.randint(0, env.action_space_n)
        
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        cumulative_rewards.append(total_reward)
        
        # Render and print
        env.render(cumulative_rewards)
        print(f"Step: {step_count}/{MAX_STEPS}, Action: {list(ACTIONS.keys())[action]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

    print("\nSimulation finished.")
    print(f"Total Reward accumulated: {total_reward:.2f}")
    
    plt.show(block=True)