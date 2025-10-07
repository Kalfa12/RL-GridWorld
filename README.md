# RL-GridWorld: A Reinforcement Learning Project

This repository contains the deliverables for a project on Reinforcement Learning, using the Grid World environment as a case study. The project explores the implementation of various RL agents, from basic environment interaction to complex, model-free learning algorithms.

## Project Structure

The project is organized into four main deliverables ("livrables"):

-   **`/livrable 1`**: Introduction to the Grid World environment. Contains a basic implementation of the environment and a simple agent that performs a random walk.
-   **`/livrable 2`**: Implementation of several foundational RL algorithms. This includes model-based agents (Value Iteration, Policy Iteration) and model-free agents (Monte Carlo, Q-Learning) that can solve the Grid World puzzle.
-   **`/livrable 3`**: Further exploration of Q-Learning with a modified agent.
-   **`/livrable 4`**: A comprehensive analysis and visualization suite. This part focuses on generating comparative performance graphs and creating animated GIFs of the trained agent's behavior.

## Livrable 4: Visualizations and Comparative Analysis

The script `livrable 4/grid_world4graphgit.py` provides a menu-driven interface to run two types of experiments and automatically saves the results.

### How to Run

1.  Clone the repository and navigate into the project folder.
2.  Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib imageio
    ```
3.  Navigate to the fourth deliverable's directory and run the script:
    ```bash
    cd "livrable 4"
    python grid_world4graphgit.py
    ```
4.  Follow the interactive menu prompts. The resulting graphs and GIFs will be saved in the `livrable 4/Results/` folder.

### Experiment 1: Single Simulation on a 5x10 Grid

This experiment trains a Q-Learning agent on a 5x10 grid with a line of obstacles and two goals with different reward values.

**Final Policy Learned by the Agent:**
The arrows show the optimal action the agent learned for each state.

![Final Policy](livrable%4/Results/single_run_policy.png)

**Trained Agent Performance (Animation):**
This GIF shows the trained agent efficiently navigating the environment to reach the highest-reward goal.

![Trained Agent GIF](livrable%204/Results/trained_agent.gif)

**Training Performance:**
These graphs illustrate the agent's learning progress by showing the total reward per episode and the decay of the exploration factor ($ \epsilon $).

| Total Rewards per Episode                                       | Epsilon Decay                                           |
| :--------------------------------------------------------------: | :------------------------------------------------------: |
| ![Rewards Plot](livrable%204/Results/single_run_rewards.png) | ![Epsilon Plot](livrable%204/Results/single_run_epsilon.png) |

---

### Experiment 2: Comparative Analysis on Different Grid Sizes

This experiment compares the learning performance of the Q-Learning agent across increasingly complex grid sizes (4x4, 6x6, and 8x8).

**Comparison of Cumulative Rewards (Smoothed):**
This plot shows that as the grid size increases, the agent requires more episodes to consistently find the optimal policy, which is expected due to the larger state space.

![Comparison Plot](livrable%204/Results/comparison_rewards.png)

**Epsilon Decay for the 8x8 Grid Run:**
This shows the epsilon decay for the final and most complex environment in the comparison analysis.

![Epsilon Decay 8x8](livrable%204/Results/last_run_epsilon_decay.png)
