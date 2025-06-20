import torch
import torch.nn as nn
import torch.optim as optim

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, env, layer_number, FLAGS, learning_rate=0.001, tau=0.05):
        super(Actor, self).__init__()
        # Determine range of actor network outputs
        if layer_number == 0:
            self.action_space_bounds = torch.tensor(env.action_bounds, dtype=torch.float32)
            self.action_offset = torch.tensor(env.action_offset, dtype=torch.float32)
        else:
            self.action_space_bounds = torch.tensor(env.subgoal_bounds_symmetric, dtype=torch.float32)
            self.action_offset = torch.tensor(env.subgoal_bounds_offset, dtype=torch.float32)

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Define the actor network
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space_size),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state, goal):
        # state: (batch, state_dim), goal: (batch, goal_dim)
        features = torch.cat([state, goal], dim=1)
        out = self.actor(features)
        # Move bounds and offset to the same device as input
        action_space_bounds = self.action_space_bounds.to(features.device)
        action_offset = self.action_offset.to(features.device)
        return (out * action_space_bounds) + action_offset

    def get_action(self, state, goal):
        # state, goal: numpy arrays
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            goal_tensor = torch.FloatTensor(goal).to(device)
            action = self.forward(state_tensor, goal_tensor)
        return action.cpu().numpy()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, target_actor):
        # Soft update model parameters.
        for target_param, param in zip(target_actor.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data) 