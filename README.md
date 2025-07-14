# RL Car Racing with DQN and Double DQN

This repository contains a Reinforcement Learning (RL) project implementing Deep Q-Networks (DQN) and Double DQN to navigate the CarRacing-v3 environment using PyTorch and Gymnasium. The project explores training agents to drive a car on random tracks, addressing challenges like runtime errors, training instability, and environment complexity.

## Project Overview

The project trains DQN and Double DQN agents on the CarRacing-v3 environment, a 2D racing game with 96x96 RGB inputs preprocessed to 84x84x4. Key components include experience replay, target networks, and soft updates (tau=0.005 for Double DQN). The goal is to compare performance, with Double DQN showing improved stability over DQN.

## Prerequisites
- Python 3.8+
- PyTorch (tested with version 2.0+)
- Gymnasium (tested with version 0.29.0+)
- NumPy
- Matplotlib

## Step-by-Step Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yayabash/rl-car-racing.git
   cd rl-car-racing

Install dependencies:

```bash
pip install torch gymnasium numpy matplotlib
```

## Project Structure
- **compare_models.py**: Script to compare DQN and Double DQN models.
- **plot_comparison.py**: Generates comparison plots (e.g., training/comparison_plot.png).
- **configs/**: Contains configuration files (e.g., dqn.yaml, double_dqn.yaml) for hyperparameters.
- **dqn_model/**: Directory for DQN model implementation and training scripts.
- **double_dqn_model/**: Directory for Double DQN model implementation and training scripts.
- **training/**: Stores logs, saved models (DQN.pt, DoubleDQN.pt), and plots.
- **videos/**: Contains video recordings of agent performance.
- **rl_project_report.pdf**: Detailed project report.

## Running the Project
- Train DQN:
  ```bash
  python dqn_model/training_dqn.py
  ```

- Train Double_DQN:
  ```bash
  python double_dqn_model/training_double_dqn.py
  ```
  
### Results

DQN: Mean reward 738.11 ± 106.88 over 1400+ episodes (356,576 steps).

Double DQN: Mean reward 619.5 ± 115.74 over 1000+ episodes (275,648 steps).

Double DQN exhibits better stability due to soft updates, though both models benefit from longer training.

## Video Visualization

## Demo Videos

≈ DQN
First trial
![DQN Episode 0](https://github.com/yayabash/rl-car-racing/blob/main/rl-video-DQN-episode-0.gif)

Second Trial
![DQN Episode 1](https://github.com/yayabash/rl-car-racing/blob/main/rl-video-DQN-episode-1.gif)

### DoubleDQN
First trial
![DoubleDQN Episode 0](https://github.com/yayabash/rl-car-racing/blob/main/rl-video-DoubleDQN-episode-0.gif)

Second Trial
![DoubleDQN Episode 1](https://github.com/yayabash/rl-car-racing/blob/main/rl-video-DoubleDQN-episode-1.gif)

## Project Report
The full project report is available as a PDF:
https://github.com/yayabash/rl-car-racing/blob/main/rl_project_report.pdf

## Future Work

Hyperparameter tuning (e.g., epsilon decay, tau).

Extend training to 100K+ timesteps.

Explore Dueling DQN or PPO for enhanced performance.

