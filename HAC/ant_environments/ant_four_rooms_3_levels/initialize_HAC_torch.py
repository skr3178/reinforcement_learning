"""
This is the starting file for the Hierarchical Actor-Critic (HAC) algorithm (PyTorch version). This script processes the command-line options specified
by the user and instantiates the environment and agent using PyTorch-based classes.
"""

from design_agent_and_env_torch import design_agent_and_env_torch
from options import parse_options
from agent_torch import AgentTorch
from run_HAC import run_HAC

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()

# Instantiate the agent and Mujoco environment using PyTorch-based agent. The designer must assign values to the hyperparameters listed in the "design_agent_and_env_torch.py" file.
agent, env = design_agent_and_env_torch(FLAGS)

# Begin training
run_HAC(FLAGS, env, agent) 