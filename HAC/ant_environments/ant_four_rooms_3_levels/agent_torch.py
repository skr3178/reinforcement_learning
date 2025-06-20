import numpy as np
import os
import pickle as cpickle
from actor_torch import Actor
from critic_torch import CriticTorch
# from layer_torch import LayerTorch  # Assume this will be created, analogous to Layer but using torch
from environment import Environment
from experience_buffer import ExperienceBuffer

class LayerTorch:
    def __init__(self, layer_number, FLAGS, env, agent_params):
        self.layer_number = layer_number
        self.FLAGS = FLAGS
        # Set time limit for each layer
        if FLAGS.layers > 1:
            self.time_limit = FLAGS.time_scale
        else:
            self.time_limit = env.max_actions
        self.current_state = None
        self.goal = None
        self.buffer_size_ceiling = 10**7
        self.episodes_to_store = agent_params["episodes_to_store"]
        self.num_replay_goals = 2
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.FLAGS.layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)
        self.batch_size = 1024
        self.replay_buffer = ExperienceBuffer(self.buffer_size, self.batch_size)
        self.temp_goal_replay_storage = []
        self.actor = Actor(env, self.layer_number, FLAGS)
        self.critic = CriticTorch(env, self.layer_number, FLAGS)
        if self.layer_number == 0:
            self.noise_perc = agent_params["atomic_noise"]
        else:
            self.noise_perc = agent_params["subgoal_noise"]
        self.maxed_out = False
        self.subgoal_penalty = agent_params["subgoal_penalty"]

    def add_noise(self, action, env):
        if self.layer_number == 0:
            action_bounds = env.action_bounds
            action_offset = env.action_offset
        else:
            action_bounds = env.subgoal_bounds_symmetric
            action_offset = env.subgoal_bounds_offset
        assert len(action) == len(action_bounds)
        assert len(action) == len(self.noise_perc)
        for i in range(len(action)):
            action[i] += np.random.normal(0, self.noise_perc[i] * action_bounds[i])
            action[i] = max(min(action[i], action_bounds[i]+action_offset[i]), -action_bounds[i]+action_offset[i])
        return action

    def get_random_action(self, env):
        if self.layer_number == 0:
            action = np.zeros((env.action_dim))
        else:
            action = np.zeros((env.subgoal_dim))
        for i in range(len(action)):
            if self.layer_number == 0:
                action[i] = np.random.uniform(-env.action_bounds[i] + env.action_offset[i], env.action_bounds[i] + env.action_offset[i])
            else:
                action[i] = np.random.uniform(env.subgoal_bounds[i][0], env.subgoal_bounds[i][1])
        return action

    def choose_action(self, agent, env, subgoal_test):
        if agent.FLAGS.test or subgoal_test:
            return self.actor.get_action(np.reshape(self.current_state, (1, len(self.current_state))), np.reshape(self.goal, (1, len(self.goal))))[0], "Policy", subgoal_test
        else:
            if np.random.random_sample() > agent.other_params["random_action_perc"]:
                action = self.add_noise(self.actor.get_action(np.reshape(self.current_state, (1, len(self.current_state))), np.reshape(self.goal, (1, len(self.goal))))[0], env)
                action_type = "Noisy Policy"
            else:
                action = self.get_random_action(env)
                action_type = "Random"
            if np.random.random_sample() < agent.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False
            return action, action_type, next_subgoal_test

    def perform_action_replay(self, hindsight_action, next_state, goal_status):
        if goal_status[self.layer_number]:
            reward = 0
            finished = True
        else:
            reward = -1
            finished = False
        transition = [self.current_state, hindsight_action, reward, next_state, self.goal, finished, None]
        self.replay_buffer.add(np.copy(transition))

    def create_prelim_goal_replay_trans(self, hindsight_action, next_state, env, total_layers):
        if self.layer_number == total_layers - 1:
            hindsight_goal = env.project_state_to_end_goal(env.sim, next_state)
        else:
            hindsight_goal = env.project_state_to_subgoal(env.sim, next_state)
        transition = [self.current_state, hindsight_action, None, next_state, None, None, hindsight_goal]
        self.temp_goal_replay_storage.append(np.copy(transition))

    def get_reward(self, new_goal, hindsight_goal, goal_thresholds):
        assert len(new_goal) == len(hindsight_goal) == len(goal_thresholds)
        for i in range(len(new_goal)):
            if np.absolute(new_goal[i] - hindsight_goal[i]) > goal_thresholds[i]:
                return -1
        return 0

    def finalize_goal_replay(self, goal_thresholds):
        num_trans = len(self.temp_goal_replay_storage)
        num_replay_goals = self.num_replay_goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans
        indices = np.zeros((num_replay_goals), dtype=int)
        if num_replay_goals > 1:
            indices[:num_replay_goals-1] = np.random.randint(num_trans, size=num_replay_goals-1)
        indices[num_replay_goals-1] = num_trans - 1
        indices = np.sort(indices)
        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)
            new_goal = trans_copy[int(indices[i])][6]
            for index in range(num_trans):
                trans_copy[index][4] = new_goal
                trans_copy[index][2] = self.get_reward(new_goal, trans_copy[index][6], goal_thresholds)
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False
                self.replay_buffer.add(trans_copy[index])
        self.temp_goal_replay_storage = []

    def penalize_subgoal(self, subgoal, next_state, high_level_goal_achieved):
        transition = [self.current_state, subgoal, self.subgoal_penalty, next_state, self.goal, True, None]
        self.replay_buffer.add(np.copy(transition))

    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True
        elif agent.steps_taken >= env.max_actions:
            return True
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True
        else:
            return False

    def train(self, agent, env, subgoal_test=False, episode_num=None):
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state
        self.maxed_out = False
        if self.layer_number == 0 and agent.FLAGS.show and agent.FLAGS.layers > 1:
            env.display_subgoals(agent.goal_array)
        attempts_made = 0
        while True:
            action, action_type, next_subgoal_test = self.choose_action(agent, env, subgoal_test)
            if self.FLAGS.Q_values:
                print("Layer %d Q-Value: " % self.layer_number, self.critic.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(action,(1,len(action)))))
                if self.layer_number == 2:
                    test_action = np.copy(action)
                    test_action[:3] = self.goal
                    print("Layer %d Goal Q-Value: " % self.layer_number, self.critic.get_Q_value(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))), np.reshape(test_action,(1,len(test_action)))))
            if self.layer_number > 0:
                agent.goal_array[self.layer_number - 1] = action
                goal_status, max_lay_achieved = agent.layers[self.layer_number - 1].train(agent, env, next_subgoal_test, episode_num)
            else:
                next_state = env.execute_action(action)
                agent.steps_taken += 1
                if agent.steps_taken >= env.max_actions:
                    print("Out of actions (Steps: %d)" % agent.steps_taken)
                agent.current_state = next_state
                goal_status, max_lay_achieved = agent.check_goals(env)
            attempts_made += 1
            if goal_status[self.layer_number]:
                if self.layer_number < agent.FLAGS.layers - 1:
                    print("SUBGOAL ACHIEVED")
                print("\nEpisode %d, Layer %d, Attempt %d Goal Achieved" % (episode_num, self.layer_number, attempts_made))
                print("Goal: ", self.goal)
                if self.layer_number == agent.FLAGS.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, agent.current_state))
            if self.layer_number == 0:
                hindsight_action = action
            else:
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                else:
                    hindsight_action = env.project_state_to_subgoal(env.sim, agent.current_state)
            if not agent.FLAGS.test:
                self.perform_action_replay(hindsight_action, agent.current_state, goal_status)
                self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, env, agent.FLAGS.layers)
                if self.layer_number > 0 and next_subgoal_test and agent.layers[self.layer_number-1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])
            if agent.FLAGS.verbose:
                print("\nEpisode %d, Level %d, Attempt %d" % (episode_num, self.layer_number, attempts_made))
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)
                if self.layer_number == agent.FLAGS.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, agent.current_state))
                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)
            self.current_state = agent.current_state
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:
                if self.layer_number == agent.FLAGS.layers-1:
                    print("HL Attempts Made: ", attempts_made)
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True
                if not agent.FLAGS.test:
                    if self.layer_number == agent.FLAGS.layers - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.subgoal_thresholds
                    self.finalize_goal_replay(goal_thresholds)
                if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
                    return goal_status, max_lay_achieved

    def learn(self, num_updates):
        for _ in range(num_updates):
            if self.replay_buffer.size > 250:
                old_states, actions, rewards, new_states, goals, is_terminals = self.replay_buffer.get_batch()
                next_batch_size = min(self.replay_buffer.size, self.replay_buffer.batch_size)
                self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_action(new_states, goals), is_terminals)
                # PyTorch actor update would require gradients from critic, which is not implemented here. You may need to adapt this for your use case.

class AgentTorch:
    def __init__(self, FLAGS, env, agent_params):
        self.FLAGS = FLAGS
        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]
        # Create agent with number of levels specified by user
        self.layers = [LayerTorch(i, FLAGS, env, agent_params) for i in range(FLAGS.layers)]
        # Model saving attributes
        self.model_dir = os.path.join(os.getcwd(), 'models_torch')
        self.model_loc = os.path.join(self.model_dir, 'HAC_torch.pt')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for _ in range(FLAGS.layers)]
        self.current_state = None
        self.steps_taken = 0
        self.num_updates = 40
        self.performance_log = []
        self.other_params = agent_params

    def check_goals(self, env):
        goal_status = [False for _ in range(self.FLAGS.layers)]
        max_lay_achieved = None
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim, self.current_state)
        for i in range(self.FLAGS.layers):
            goal_achieved = True
            if i == self.FLAGS.layers - 1:
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds)
                for j in range(len(proj_end_goal)):
                    if np.abs(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break
            else:
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.subgoal_thresholds)
                for j in range(len(proj_subgoal)):
                    if np.abs(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False
        return goal_status, max_lay_achieved

    def save_model(self, episode):
        # Save all layer actor/critic weights
        torch_save_dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'actor') and hasattr(layer, 'critic'):
                torch_save_dict[f'actor_{i}'] = layer.actor.state_dict()
                torch_save_dict[f'critic_{i}'] = layer.critic.critic.state_dict()
        import torch
        torch.save(torch_save_dict, self.model_loc + f'_ep{episode}.pt')

    def learn(self):
        for i in range(len(self.layers)):
            self.layers[i].learn(self.num_updates)

    def train(self, env, episode_num, total_episodes):
        self.goal_array[self.FLAGS.layers - 1] = env.get_next_goal(self.FLAGS.test)
        print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])
        self.current_state = env.reset_sim(self.goal_array[self.FLAGS.layers - 1])
        if env.name == "ant_reacher.xml":
            print("Initial Ant Position: ", self.current_state[:3])
        self.steps_taken = 0
        goal_status, max_lay_achieved = self.layers[self.FLAGS.layers-1].train(self, env, episode_num=episode_num)
        if not self.FLAGS.test and total_episodes > self.other_params["num_pre_training_episodes"]:
            self.learn()
        return goal_status[self.FLAGS.layers-1]

    def log_performance(self, success_rate):
        self.performance_log.append(success_rate)
        cpickle.dump(self.performance_log, open("performance_log_torch.p", "wb")) 