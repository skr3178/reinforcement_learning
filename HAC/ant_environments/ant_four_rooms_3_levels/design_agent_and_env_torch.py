"""
This file provides the template for designing the agent and environment (PyTorch version). The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np
from environment import Environment
from utils import check_validity
from agent_torch import AgentTorch

def design_agent_and_env_torch(FLAGS):
    """
    1. DESIGN AGENT
    The key hyperparameters for agent construction are
        a. Number of levels in agent hierarchy
        b. Max sequence length in which each policy will specialize
        c. Max number of atomic actions allowed in an episode
        d. Environment timesteps per atomic action
    See Section 3 of this file for other agent hyperparameters that can be configured.
    """
    FLAGS.layers = 3    # Enter number of levels in agent hierarchy
    FLAGS.time_scale = 10    # Enter max sequence length in which each policy will specialize
    # Enter max number of atomic actions.  This will typically be FLAGS.time_scale**(FLAGS.layers).  However, in the UR5 Reacher task, we use a shorter episode length.
    max_actions = 700
    # max_actions = 15
    timesteps_per_action = 15    # Provide the number of time steps per atomic action.
    """
    2. DESIGN ENVIRONMENT
    """
    model_name = "ant_four_rooms.xml"
    initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
    initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
    initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
    initial_joint_ranges[0] = np.array([-6,6])
    initial_joint_ranges[1] = np.array([-6,6])
    initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)
    max_range = 6
    goal_space_train = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]]
    goal_space_test = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]]
    project_state_to_end_goal = lambda sim, state: state[:3]
    len_threshold = 0.4
    height_threshold = 0.2
    end_goal_thresholds = np.array([len_threshold, len_threshold, height_threshold])
    cage_max_dim = 8
    max_height = 1
    max_velo = 3
    subgoal_bounds = np.array([[-cage_max_dim,cage_max_dim],[-cage_max_dim,cage_max_dim],[0,max_height],[-max_velo, max_velo],[-max_velo, max_velo]])
    project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))
    velo_threshold = 0.8
    quat_threshold = 0.5
    subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, velo_threshold, velo_threshold])
    agent_params = {}
    agent_params["subgoal_test_perc"] = 0.3
    agent_params["subgoal_penalty"] = -FLAGS.time_scale
    agent_params["random_action_perc"] = 0.3
    agent_params["atomic_noise"] = [0.2 for i in range(8)]
    agent_params["subgoal_noise"] = [0.2 for i in range(len(subgoal_thresholds))]
    agent_params["num_pre_training_episodes"] = 30
    agent_params["episodes_to_store"] = 500
    agent_params["num_exploration_episodes"] = 100
    check_validity(model_name, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action)
    env = Environment(model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions, timesteps_per_action, FLAGS.show)
    agent = AgentTorch(FLAGS, env, agent_params)
    return agent, env 