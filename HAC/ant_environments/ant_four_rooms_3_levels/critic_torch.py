import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CriticTorch:
    def __init__(self, env, layer_number, FLAGS, learning_rate=0.001, gamma=0.98, tau=0.05):
        self.critic_name = 'critic_' + str(layer_number)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        
        self.q_limit = -FLAGS.time_scale

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            self.action_dim = env.action_dim
        else:
            self.action_dim = env.subgoal_dim

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

        # Target network
        self.target_critic = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.mseLoss = nn.MSELoss()

    def get_Q_value(self, state, goal, action):
        # state, goal, action: np arrays, shape (batch, dim)
        state = torch.FloatTensor(state).to(device)
        goal = torch.FloatTensor(goal).to(device)
        action = torch.FloatTensor(action).to(device)
        features = torch.cat([state, goal, action], dim=1)
        with torch.no_grad():
            q = -self.critic(features) * self.q_limit
        return q.cpu().numpy()

    def get_target_Q_value(self, state, goal, action):
        state = torch.FloatTensor(state).to(device)
        goal = torch.FloatTensor(goal).to(device)
        action = torch.FloatTensor(action).to(device)
        features = torch.cat([state, goal, action], dim=1)
        with torch.no_grad():
            q = -self.target_critic(features) * self.q_limit
        return q.cpu().numpy()

    def update(self, old_states, old_actions, rewards, new_states, goals, new_actions, is_terminals):
        # Convert to tensors
        old_states = torch.FloatTensor(old_states).to(device)
        old_actions = torch.FloatTensor(old_actions).to(device)
        rewards = torch.FloatTensor(rewards).reshape((-1,1)).to(device)
        new_states = torch.FloatTensor(new_states).to(device)
        goals = torch.FloatTensor(goals).to(device)
        new_actions = torch.FloatTensor(new_actions).to(device)
        is_terminals = torch.FloatTensor(is_terminals).reshape((-1,1)).to(device)

        # Compute target Q values
        with torch.no_grad():
            next_features = torch.cat([new_states, goals, new_actions], dim=1)
            wanted_qs = -self.critic(next_features) * self.q_limit
            # Uncomment to use target network:
            # wanted_qs = -self.target_critic(next_features) * self.q_limit

        # Bellman update
        target_qs = rewards.clone()
        not_done = (is_terminals == 0)
        target_qs[not_done] += self.gamma * wanted_qs[not_done]
        # Clamp Q target to [-self.time_limit, 0]
        target_qs = torch.clamp(target_qs, min=self.q_limit, max=0)

        features = torch.cat([old_states, goals, old_actions], dim=1)
        current_qs = -self.critic(features) * self.q_limit
        loss = self.mseLoss(current_qs, target_qs)
        self.loss_val = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_weights(self):
        # Polyak averaging
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_gradients(self, state, goal, action):
        # Returns gradients of Q w.r.t. action
        state = torch.FloatTensor(state).to(device)
        goal = torch.FloatTensor(goal).to(device)
        action = torch.FloatTensor(action).to(device)
        state.requires_grad = False
        goal.requires_grad = False
        action.requires_grad = True
        features = torch.cat([state, goal, action], dim=1)
        q = -self.critic(features) * self.q_limit
        grads = torch.autograd.grad(q.sum(), action, retain_graph=True)[0]
        return grads.cpu().numpy() 