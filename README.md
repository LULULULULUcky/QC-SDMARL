Quick Start Guide

# Prerequisites
Before you begin, ensure you have the following installed:
Python 3.9​ (Required)
pip​ (Python package manager)
Git​ (for cloning repository)
Conda​ (recommended for environment management, but virtualenv also works)

Step 1: Get the Code
Option A: Clone via Git (Recommended)
Option B: Download ZIP
Click the green "Code" button on GitHub
Select "Download ZIP"
Extract the archive to your desired location

Step 2: Set Up Python Environment
Using Conda or Using venv
Create a new conda environment with Python 3.9: conda create -n underwater-marl python=3.9
Activate the environment: conda activate underwater-marl

Step 3: Install Dependencies
Install all required packages: pip install -r requirements.txt

Step 4: Run the Project
run the training script directly: train.py

# Detailed Code File Documentation

1. Core Algorithm Modules

DDPG.py
Function: Implements the single-agent Deep Deterministic Policy Gradient algorithm.
Key Classes:
TwoLayerFC: Actor network structure (2-layer fully connected network).
Critic: Critic network that implements a dynamic attention mechanism.
DDPG: Includes basic functions such as network initialization and action selection.

MADDPG.py
Function: Implements the multi-agent DDPG framework.
Features: Handles collaboration mechanisms between multiple agents, including contribution-based experience sampling, network updates, and parameter synchronization.

QMADDPG.py
Function: The core innovation of this project.
Features: Implements Quantum Computing Optimization (QCO)-driven weight allocation and QCO-driven experience replay buffer updates.

2. Environment Simulation Module

environment.py
Function: Defines environment entities (such as agents and targets) and their physical simulation.
Details: Initializes agent and target classes, implements fundamental world rules, including entity class definitions, fluid dynamics calculations (drag/lift/Morison forces), collision detection, and position updates.

make_world.py
Function: Core module for constructing the multi-agent underwater tracking and encirclement scenario.
Details: Responsible for scene construction and initialization, including agent grouping, target allocation mechanisms, and sonar detection implementation.
Note: If modifications to the encirclement strategy or formation count are required, adjust relevant parameters in this file.

multienvironment.py
Function: Provides 2D visualization functionality.
Details: Used to display the tracking process in a 2D visual interface.

3. Utilities and Helper Modules

load.py
Function: Initial formation allocation mechanism.
Details: Stores the logic for initial formation allocation.

misc.py
Function: Low-level utility functions.
Details: Contains some general low-level helper functions.

quantum_utils.py
Function: Quantum computing-related tools.
Details: Includes auxiliary functionalities such as quantum annealing for experience updates.

rl_utils.py
Function: General reinforcement learning utilities.
Details: Contains functions like experience replay buffer and data smoothing.

tool_functions.py
Function: General utility functions.
Details: Contains relatively low-level code snippets.

4. Main Program Module

train.py
Function: Main training script. 
Details: include training the main loop, such as resampling core logic. 
Note: The parts commented out at the end of the file are various visualization codes.

# Parameter Tuning Guide
If you need to adjust the number of agents or targets, please refer to the following modification points:
1. Modify Agent and Target Counts:
Adjust the number of agents in both train.py​ and make_world.py.
Adjust the number of targets in make_world.py.
2. Modify Formation Size:
In make_world.py, locate the line:
for i in range(3, 36, 3):
Rule: The second number (currently 36) represents the total number of agents. When modifying, follow the rule: agent count × 3.
3. Modify Visualization Parameters:
Adjust the three relevant parameters at the top of multienvironment.py.

