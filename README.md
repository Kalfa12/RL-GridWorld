# RL-GridWorld: A Reinforcement Learning Project

This repository contains the deliverables for a project on Reinforcement Learning, using the Grid World environment as a case study. The project explores the implementation of various RL agents, from basic environment interaction to advanced, deep learning-based agents capable of generalization.

## Project Structure

The project is organized into five main deliverables ("livrables"):

-   **/livrable 1**: Introduction to the Grid World environment. Contains a basic implementation of the environment and a simple agent that performs a random walk.
-   **/livrable 2**: Implementation of several foundational RL algorithms. This includes model-based agents (Value Iteration, Policy Iteration) and model-free agents (Monte Carlo, Q-Learning).
-   **/livrable 3**: Further exploration and modification of the Q-Learning algorithm.
-   **/livrable 4**: A comprehensive analysis and visualization suite for the standard Q-Learning agent, including comparative performance graphs and animations.
-   **/livrable 5**: An advanced implementation of a Deep Q-Learning (DQN) agent designed to solve a more complex, dynamic environment.

---

## Livrable 4: Q-Learning Agent with Visualizations

The script `livrable 4/grid_world4graphgit.py` provides a menu-driven interface to run two types of experiments for a Q-Learning agent in a static environment with fixed goals and obstacles. All results are saved to the `livrable 4/Results/` folder.

### How to Run

1.  Navigate to the fourth deliverable's directory.
2.  Run the script: `python grid_world4graphgit.py`
3.  Follow the interactive menu to choose between a single, detailed simulation (option 1) or a comparative analysis across different grid sizes (option 2).

### Experiment 1: Single Simulation on a 5x10 Grid

This experiment trains a Q-Learning agent on a specific 5x10 grid with obstacles.

**Final Policy Learned by the Agent:**

![Final Policy](livrable%204/Results/single_run_policy.png)

**Agent Performance (Animation):**

![Trained Agent GIF](livrable%204/Results/trained_agent.gif)

### Experiment 2: Comparative Analysis on Different Grid Sizes

This experiment compares the agent's learning performance across different square grid sizes.

**Comparison of Cumulative Rewards (Smoothed):**

![Comparison Plot](livrable%204/Results/comparison_rewards.png)

---

## Livrable 5: Deep Q-Learning for a Dynamic Environment

This deliverable addresses a key limitation of traditional RL methods. We introduce an environment where the **goal moves to a new, random position at the start of every episode.**

### The Challenge: Why Standard Q-Learning Fails

A standard Q-table agent learns a specific value for every state. In a dynamic environment, the "state" must include both the agent's position and the goal's position. This leads to two critical problems:
1.  **State-Space Explosion:** The number of possible states becomes enormous, making the Q-table impractically large and slow to train.
2.  **Failure to Generalize:** The agent only learns a rigid path for each specific goal location it has seen. If the goal appears in a new, unseen position, the agent has no prior knowledge and cannot infer a correct policy. It cannot understand the *concept* of "moving towards the goal."

### The Solution: Deep Q-Learning (DQN)

To solve this, we use a **Deep Q-Network (DQN)**. Instead of a table, we use a neural network as a function approximator. The network takes the state `[agent_row, agent_col, goal_row, goal_col]` as input and *estimates* the Q-values for each possible action. This allows the agent to **generalize** its knowledge across different goal locations, learning a robust policy that works for any goal position.

### How to Run

The script for this deliverable saves and loads the trained model to avoid re-training.
1.  Navigate to the fifth deliverable's directory.
2.  Run the script: `python grid_worldDQL.py`.
3.  The first time, it will train the model for 500 episodes and save all results to the `livrable 5/results.deepqlearning.dql/` folder. Subsequent runs will load the saved model and proceed directly to the visual evaluation.

### Results and Interpretation

The DQN agent successfully learns a general policy to solve the dynamic grid world problem.

**Agent Performance (Animation)**
This GIF shows the trained agent's performance on a random task. The agent efficiently navigates from the start (top-left) to the randomly placed goal (gold square), proving it has learned a general strategy.

![Agent Performance](livrable%205/results.deepqlearning.dql/dqn_agent_evaluation.gif)

**Training Performance**
These graphs illustrate the agent's learning process. The upward trend in rewards and the downward trend in loss are signs of a healthy and successful training session.

| Total Rewards per Episode                                       | Training Loss                                           |
| :--------------------------------------------------------------: | :------------------------------------------------------: |
| ![Rewards Plot](livrable%205/results.deepqlearning.dql/rewards_per_episode.png) | ![Loss Plot](livrable%205/results.deepqlearning.dql/training_loss.png) |
