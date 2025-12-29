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
    %% Style Definitions (Soft Colors)
    classDef state fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,rx:10,ry:10;
    classDef action fill:#ffebee,stroke:#c62828,stroke-width:2px,rx:10,ry:10;
    classDef reward fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;
    classDef memory fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,shape:cylinder;
    classDef network fill:#fff8e1,stroke:#fbc02d,stroke-width:2px,rx:5,ry:5;

    %% --- 1. INTERACTION PHASE ---
    subgraph INTERACTION [" 🌍 Robot-Environment Loop "]
        direction LR
        State["📷 Observation"]:::state
        Action["🦾 Action"]:::action
        Env("🏗️ Simulation<br>(Robosuite)"):::reward
        
        State -->|Input| Action
        Action -->|Execute| Env
        Env -->|Feedback| State
    end

    %% --- 2. STORAGE PHASE ---
    subgraph MEMORY [" 🧠 Experience Replay "]
        Buffer[("🗄️ Replay Buffer<br>(1M Samples)")]:::memory
    end

    %% --- 3. TRAINING PHASE ---
    subgraph LEARNING [" 🎓 Optimization (TD3) "]
        direction TB
        
        subgraph CRITIC_UPDATE [" ⚖️ Critic Training "]
            Batch1(Sample Batch):::network
            Critic["📉 Twin Critics<br>(Minimize Error)"]:::network
        end
        
        subgraph ACTOR_UPDATE [" 🎭 Actor Training "]
            Actor["📈 Actor Network<br>(Maximize Reward)"]:::network
        end
    end

    %% --- CONNECTIONS ---
    
    %% Storage
    Env -.->|"(s, a, r, s')"| Buffer
    
    %% Learning Data Flow
    Buffer ==>|Random Batch| Batch1
    Batch1 --> Critic
    Batch1 --> Actor
    
    %% Gradients
    Actor -->|Policy| Critic
    
    %% Target Networks (Implicit)
    Critic -.->|Polyak Avg| Target[Target Networks]:::network
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
