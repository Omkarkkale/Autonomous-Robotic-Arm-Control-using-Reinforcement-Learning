<div align="center">

# 🤖 Autonomous Robotic Arm Control
### using Reinforcement Learning (TD3)

![Python](https://img.shields.io/badge/Python-3.10.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Robosuite](https://img.shields.io/badge/Sim-Robosuite-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

<br>

![Demo](assets/demo.gif)

**Training a Franka Emika Panda robot to open a door using the Twin Delayed DDPG (TD3) algorithm.**

</div>

---

## 📖 Overview

This project demonstrates the power of **Deep Reinforcement Learning** in continuous control robotics. Using the **Robosuite** simulation framework (powered by MuJoCo), we trained an agent to solve the complex `Door` task from scratch.

The agent operates in a continuous state and action space, learning to coordinate 7 degrees of freedom to approach, grasp, rotate, and push the door handle.

## 🚀 Key Features

-   **🧠 Advanced RL:** Custom implementation of **TD3** (Twin Delayed Deep Deterministic Policy Gradient), improving over DDPG by reducing overestimation bias.
-   **🦾 High-Fidelity Sim:** Built on **Robosuite**, offering realistic physics and collisions.
-   **📈 Proven Convergence:** Solved the environment with a stable high score of **~275**.
-   **🛡️ Robustness:** Includes mechanisms for **Checkpointing** and **Best Model Preservation**.

---

## 📊 Performance

The agent was trained for **8,500+ episodes**. It demonstrates a clear "S-curve" learning trajectory, mastering the task after an initial exploration phase.

<div align="center">
  <img src="assets/training_graph.png" width="800" alt="Training Graph">
</div>

---

## 🛠️ Installation

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/Omkarkkale/Autonomous-Robotic-Arm-Control-using-Reinforcement-Learning.git
    cd Autonomous-Robotic-Arm-Control-using-Reinforcement-Learning
    ```

2.  **Setup Environment** (Recommended):
    *Requires Python 3.10.8*
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/Mac
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🕹️ Usage

### 👁️ Watch the AI (Visualization)
To see the trained agent in action:
```bash
python test.py
```
*Loads the best weights and renders the simulation.*

### 🏋️ Train from Scratch
To start a new training session:
```bash
python main.py
```
*Logs are saved to `logs/` (viewable with TensorBoard).*

---

## 🧠 Model Architecture (TD3)

The solution uses an **Actor-Critic** architecture with **Twin Delayed** stabilization:

```mermaid
graph TD
    %% Define Styles
    classDef storage fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef network fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef environment fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    subgraph Simulation [Environment Loop]
        direction TB
        Env((Robosuite<br>Environment)):::environment
        State[Observation]:::process
        Reward(Reward):::process
        Done{Done?}:::process
    end

    subgraph Training [TD3 Learning Process]
        direction TB
        Buffer[("Replay Buffer<br>(1,000,000 capacity)")]:::storage
        
        subgraph Critics [Critic Update]
            Batch(Sample Mini-Batch):::process
            Critic1[Critic 1 Network]:::network
            Critic2[Critic 2 Network]:::network
            TargetC1[Target Critic 1]:::network
            TargetC2[Target Critic 2]:::network
            LossC(Minimize  MSE Loss):::process
        end

        subgraph Actors [Delayed Actor Update]
            Actor[Actor Network]:::network
            TargetActor[Target Actor]:::network
            LossA(Maximize Q-Value):::process
        end
    end

    %% Data Flow
    Env -->|State| Actor
    Actor -->|Action + Noise| Env
    Env -->|Next State, Reward, Done| Buffer
    
    %% Critic Training Flow
    Buffer -->|Batch| Batch
    Batch --> Critic1 & Critic2
    Batch --> TargetC1 & TargetC2 & TargetActor
    TargetActor -->|Target Action| TargetC1 & TargetC2
    TargetC1 & TargetC2 -->|Min Q-Value| LossC
    LossC -->|Backprop| Critic1 & Critic2

    %% Actor Training Flow
    Batch --> Actor
    Actor -->|Predicted Action| Critic1
    Critic1 -->|Q-Value| LossA
    LossA -->|Backprop| Actor
    
    %% Target Updates
    Actor -.->|Polyak Avg| TargetActor
    Critic1 -.->|Polyak Avg| TargetC1
    Critic2 -.->|Polyak Avg| TargetC2
```

*   **Actor:** Maps states to continuous actions (Joint Velocities).
*   **Critic (x2):** Estimates the Q-value of state-action pairs (Twin Critics to reduce bias).
*   **Target Networks:** Normalized using Polyak averaging for stability.

---

## 🔗 References
*   [TD3 Paper (Fujimoto et al.)](https://arxiv.org/abs/1802.09477)
*   [Robosuite Documentation](https://robosuite.ai/)

<div align="center">
  <sub>Built by Omkar Kale</sub>
</div>
