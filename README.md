# Autonomous Robotic Arm Control using Reinforcement Learning (TD3)

This project implements a **Twin Delayed Deep Deterministic Policy Gradient (TD3)** agent to control a robotic arm (Franka Emika Panda) in the **Robosuite** simulation environment. The specific task is **Door Opening**.

## Project Overview

The goal of this project is to train an autonomous agent to manipulate a robotic arm to approach, grasp, and open a door using raw sensor observations. The project demonstrates the successful application of Deep Reinforcement Learning (DRL) for continuous control in robotics.

### Key Features
*   **Algorithm:** Custom implementation of TD3 (Twin Delayed DDPG) in PyTorch.
*   **Environment:** [Robosuite](https://robosuite.ai/) (MuJoCo physics engine).
*   **Task:** `Door` (Panda robot).
*   **State Space:** Joint positions, velocities, end-effector pose, and object information.
*   **Action Space:** Continuous control (Joint velocities / Torques).
*   **Performance:** The agent achieves a convergence score of ~275, successfully opening the door consistently.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Omkarkkale/Autonomous-Robotic-Arm-Control-using-Reinforcement-Learning.git
    cd Autonomous-Robotic-Arm-Control-using-Reinforcement-Learning
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    You need strict version compatibility for MuJoCo and Robosuite.
    ```bash
    pip install numpy==1.26.4
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
    pip install robosuite
    pip install gym==0.26.2
    pip install tensorboard
    ```
    *Note: Hardware acceleration (CUDA) is highly recommended for training.*

## Usage

### 1. Training the Agent
To train the agent from scratch (or resume from a checkpoint):
```bash
python main.py
```
*   **Checkpoints:** Models are saved in `tmp/td3/` every 10 episodes.
*   **Best Model:** The best performing model is automatically saved to `tmp/td3/best/`.
*   **Monitoring:** Use TensorBoard to view training progress.
    ```bash
    tensorboard --logdir=logs
    ```

### 2. Testing / Visualization
To watch the trained agent perform the task:
```bash
python test.py
```
This loads the trained weights and renders the simulation on-screen.

## Project Structure

*   `main.py`: The entry point for training. Initializes the environment and the training loop.
*   `test.py`: Script for visualizing the trained agent's performance.
*   `td3_torch.py`: Core implementation of the TD3 Agent (Actor-Critic architecture).
*   `networks.py`: PyTorch definitions for the Actor and Critic neural networks.
*   `buffer.py`: Replay Buffer implementation for experience replay.
*   `checkpoint.txt`: Tracks the current episode number for resuming training.

## Results

The agent successfully converges after approximately 8,000 episodes, achieving a stable high score of ~275.
*(Include screenshots or GIFs of the robot opening the door here)*

## References
*   [TD3 Paper: Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
*   [Robosuite Documentation](https://robosuite.ai/docs/overview.html)
