"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent.
"""

from design_agent_and_env import design_agent_and_env
from options import parse_options
from agent import Agent
from run_HAC import run_HAC
import tensorflow as tf

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()

# Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
agent, env = design_agent_and_env(FLAGS)

# Begin training
run_HAC(FLAGS,env,agent)

config = tf.ConfigProto(
    device_count={'CPU': FLAGS.num_cpus},
    inter_op_parallelism_threads=FLAGS.num_cpus,
    intra_op_parallelism_threads=FLAGS.num_cpus
)
agent.sess = tf.Session(config=config)
