RL Car Racing with DQN and Double DQN
This repository contains a Reinforcement Learning (RL) project implementing Deep Q-Networks (DQN) and Double DQN to navigate the CarRacing-v3 environment using PyTorch and Gymnasium. The project explores training agents to drive a car on random tracks, addressing challenges like runtime errors, training instability, and environment complexity.
Project Overview
The project trains DQN and Double DQN agents on the CarRacing-v3 environment, a 2D racing game with 96x96 RGB inputs preprocessed to 84x84x4. Key components include experience replay, target networks, and soft updates (tau=0.005 for Double DQN). The goal is to compare performance, with Double DQN showing improved stability over DQN.
Files

compare_models.py: Script to compare DQN and Double DQN models.
plot_comparison.py: Generates comparison plots (e.g., training/comparison_plot.png).
configs/dqn.yaml, configs/double_dqn.yaml: Hyperparameter configurations.
double_dqn_model/, dqn_model/: Model implementations and training scripts.
training/: Contains logs, saved models (DQN.pt, DoubleDQN.pt), and plots.
videos/DQN/, videos/DoubleDQN/: Video recordings of agent performance.
rl_project_report.pdf: Detailed project report (see below).

Setup

Clone the repository:git clone https://github.com/yayabash/rl-car-racing.git
cd rl-car-racing


Install dependencies:pip install torch gymnasium numpy matplotlib


Run training scripts:python dqn_model/training_dqn.py
python double_dqn_model/training_double_dqn.py



Results

DQN: Mean reward 738.11 ± 106.88 over 1400+ episodes (356,576 steps).
Double DQN: Mean reward 619.5 ± 115.74 over 1000+ episodes (275,648 steps).
Double DQN exhibits better stability due to soft updates, though both models benefit from longer training.

Video Visualization
Watch the agents in action:

DQN Episode 0
DQN Episode 1
Double DQN Episode 0
Double DQN Episode 1

Project Report
The full project report is available as a PDF:rl_project_report.pdf
Future Work

Hyperparameter tuning (e.g., epsilon decay, tau).
Extend training to 100K+ timesteps.
Explore Dueling DQN or PPO for enhanced performance.

