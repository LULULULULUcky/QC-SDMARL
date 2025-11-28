
First of all, install the packages as required by requirements.txt.
This code package mainly consists of the following several parts
DDPG:
    TwoLayerFC class: Actor network structure (2-layer fully connected)
    Critic class: A Critic network that implements a dynamic attention mechanism
    DDPG class: It includes basic functions such as network initialization and action selection
MADDPG:
    The multi-agent DDPG framework completes the multi-agent collaboration mechanism, experience sampling based on contribution, network update and parameter synchronization
QMADDPG:
    The innovation point of QC-SDMARL is the implementation of quantum-enhanced MADDPG
    Including QCO-driven weight allocation and QCO-driven experience replay buffer update
environment:
    Define environmental entities and physical simulations. 
    Initialize classes such as agents and targets, and then implement some basic rules of the world
    Including entity class definitions (Agent/Target, etc.), fluid dynamics calculations (drag/lift /Morison force), collision detection and position updates
load:
    The content stored here is the initial formation allocation mechanism.
make_world:
    The core environmental building block for multi-agent underwater tracking and encirclement
    Realize scene construction and initialization, including agent grouping and target allocation mechanisms, as well as sonar detection implementation
    Reward Function Design (Implementing the Encirclement Strategy)
    Note: To modify the formation number, you need to adjust group_size
misc:
    Relatively low-level functions
multienvironment:
    It includes functions such as two-dimensional visual display of the tracking process
N-S:
    It has not been called and has been implemented in the environment code file
rl_utils:
    Experience playback buffer, data smoothing processing, quantum annealing experience update
tool_functions:
    It is relatively low-level code
train:
    The main function can be run by running this code file when running the code. 
    It includes the main code such as training resampling, etc. 
    The parts commented out later are various visual codes

In addition, if it is necessary to change the number of agents and targets, the following contents can be modified:

The number of agents of train and makeword plus the target number of the latter

The formation number in makeword and for i in range(3, 36, 3):# Change the second number within the number of agents (this is 12 agents), the rule is: number of agents *3
Change the top three quantities in mutienvironment