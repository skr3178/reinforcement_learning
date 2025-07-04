#recurrent will help the agent rememeber the previous ordeal
import collections
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
from ma_gym.wrappers import Monitor

USE_WANDB = False  # if enabled, logs data on wandb server

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size) # start_idx =32, len(buffer) = 2014
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0])
        return torch.tensor(s_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1)

    def size(self):
        return len(self.buffer)

# Combines those individual Q-values into a single global Q-value in a way
# that supports centralized training.
class MixNet(nn.Module):
    # returns x(all applied), hidden
    # input: obs_space, hidden_dim, hx_size
    # output: x[state updated with NN operations], hidden
    def __init__(self, observation_space, hidden_dim=32, hx_size=64, recurrent=False):
        super(MixNet, self).__init__()
        state_size = sum([_.shape[0] for _ in observation_space]) #94
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        self.n_agents = len(observation_space)
        self.recurrent = recurrent

        hyper_net_input_size = state_size
        if self.recurrent:
            self.gru = nn.GRUCell(state_size, self.hx_size) # a hidden learning state for memory based training
            hyper_net_input_size = self.hx_size # 64

        self.hyper_net_weight_1 = nn.Linear(hyper_net_input_size, self.n_agents * hidden_dim) #[64, 2*32] input to the weight is the states, output:  Q_values * hidden dimension
        self.hyper_net_weight_2 = nn.Linear(hyper_net_input_size, hidden_dim) #[94, 32] state size and output is hidden dimension

        self.hyper_net_bias_1 = nn.Linear(hyper_net_input_size, hidden_dim) #[94, 32]
        self.hyper_net_bias_2 = nn.Sequential(nn.Linear(hyper_net_input_size, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, 1))

    def forward(self, q_values, observations, hidden): # q_values[32, 2],  s[:, step_i, :, :]=obs[32, 2, 47]
        batch_size, n_agents, obs_size = observations.shape
        state = observations.view(batch_size, n_agents * obs_size) # matrix operations can only be done on 2D tensors

        x = state #[32, 64]
        if self.recurrent:
            hidden = self.gru(x, hidden)
            x = hidden
        weight_1 = torch.abs(self.hyper_net_weight_1(x)) #[32 x 64] # weights are positive only
        weight_1 = weight_1.view(batch_size, self.hidden_dim, n_agents) #[32, 32, 2]# breaking it into batches
        bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1) # [32, 32, 1]
        weight_2 = torch.abs(self.hyper_net_weight_2(x)) # [32, 32]
        bias_2 = self.hyper_net_bias_2(x) #[32, 1]

        x = torch.bmm(weight_1, q_values.unsqueeze(-1)) + bias_1 # torch.Size([32, 32, 2]), torch.Size([32, 2, 1]), torch.Size([32, 2, 1])
        x = torch.relu(x)
        x = (weight_2.unsqueeze(-1) * x).sum(dim=1) + bias_2
        return x, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.hx_size))

# Learns the individual Q-values for each agent based on local observation
class QNet(nn.Module):
    # inputL obs_space, action_space
    # Output:torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space) # =2, Box[47,],Box[47,]
        self.recurrent = recurrent
        self.hx_size = 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0] #47
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                            nn.ReLU(),
                                                                            nn.Linear(128, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size, )] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden) # out[1, 2, 5] hidden[1, 2, 32]
        mask = (torch.rand((out.shape[0],)) <= epsilon) # Tensor[True]
        action = torch.empty((out.shape[0], out.shape[1],)) # [0, 0]
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float() # torch.size([47,5])
        if (~mask).any(): # # Explore state-space?? if conditional argument comparing the torch.rand to epsilon is False
            action[~mask] = out[~mask].argmax(dim=2).float() # Q-learning standard: step1- compute the Q(s,a) table. step2- pick the action with the highest Q(s,a)
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))


def train(q, q_target, mix_net, mix_net_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10,
          grad_clip_norm=5):
    _chunk_size = chunk_size if q.recurrent else 1
    for _ in range(update_iter):
        s, a, r, s_prime, done = memory.sample_chunk(batch_size, _chunk_size)
        hidden = q.init_hidden(batch_size) # zeros [32, 2,  32]
        target_hidden = q_target.init_hidden(batch_size) #[32, 2, 32]
        mix_net_target_hidden = mix_net_target.init_hidden(batch_size) #[32, 64]
        mix_net_hidden = [torch.empty_like(mix_net_target_hidden) for _ in range(_chunk_size + 1)] # 11 x size[32, 64]
        # mix_net_hidden[0] = mix_net_target.init_hidden(batch_size)
        loss = 0

        for step_i in range(_chunk_size):
            # s is all the states[32, 10, 2, 47]--> [32, 2, 47]  # q_out: [32, 2, 5], hidden: shape[32, 2, 32]
            # predicted Q-values for all 5 actions for each agent
            q_out, hidden = q(s[:, step_i, :, :], hidden)

            # extract the Q-values for the actions taken by each agent at that step
            # q_out: shape [batch_size, n_agents, n_actions]
            # → Output of q(s, hidden), i.e., Q-values predicted for all actions of all agents at the current state.
            # a[:, step_i, :]: shape [batch_size, n_agents]
            # → The actual actions taken by each agent at this timestep
            #q_a [32, 2], q_out [32, 2, 5]
            q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)#

            # predict the mixed Q-value for the current state
            # q_a undergoes NN operations, s
            # pred_q [32, 1], next_mix_net_hidden [32, 64]
            pred_q, next_mix_net_hidden = mix_net(q_a, s[:, step_i, :, :], mix_net_hidden[step_i]) #q_a: [32, 1], s(step_i):[32, 2, 47]

            # Get Q-values for next-state using target Q-network
            # Computes Q-values for each action for each agent at the next state
            max_q_prime, target_hidden = q_target(s_prime[:, step_i, :, :], target_hidden.detach()) #s'[:,step_i, :, :]:

            # max Individual Q-values for each agent at that step or best action's Q-value for each agent at that step
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1) #[batch_size, n_agents]

            q_prime_total, mix_net_target_hidden = mix_net_target(max_q_prime, s_prime[:, step_i, :, :], mix_net_target_hidden.detach())
            target_q = r[:, step_i, :].sum(dim=1, keepdims=True) + (gamma * q_prime_total * (1 - done[:, step_i])) #target_q: [32, 1]

            loss += F.smooth_l1_loss(pred_q, target_q.detach())
            # print(pred_q)
            # print(target_q)
            # resetting the hidden states of the recurrent networks when an episode ends 
            # done_mask = done[:, step_i].squeeze(-1).bool() #[32, 10, 1]
            # hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))
            # target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))
            #
            # mix_net_hidden[step_i + 1][~done_mask] = next_mix_net_hidden[~done_mask]
            # mix_net_hidden[step_i + 1][done_mask] = mix_net.init_hidden(len(mix_net_hidden[step_i][done_mask]))
            # mix_net_target_hidden[done_mask] = mix_net_target.init_hidden(len(mix_net_target_hidden[done_mask]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(mix_net.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()


def evaluate_model(env, num_episodes, q):
    score = 0
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            while not all(done):
                env.render()
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
                next_state, reward, done, info = env.step(action[0].data.cpu().numpy().tolist())
                score += sum(reward)
                state = next_state

    return score / num_episodes


def main(env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter, chunk_size,
         update_target_interval, recurrent, record):
    # create env.
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    if record:
        if os.path.exists('recordings'):
            shutil.rmtree('recordings')
        env = Monitor(env, directory='recordings', force=True, video_callable=lambda episode_id: episode_id % 50 == 0)
        test_env = Monitor(test_env, directory='recordings', force=True, video_callable=lambda episode_id: True)

    memory = ReplayBuffer(buffer_limit)

    # create networks
    q = QNet(env.observation_space, env.action_space, recurrent) # Obs_space: [Box(47,), Box(47,)], Act_space: [Discrete(5), Discrete(5)]
    q_target = QNet(env.observation_space, env.action_space, recurrent)
    q_target.load_state_dict(q.state_dict())

    mix_net = MixNet(env.observation_space, recurrent=recurrent)
    mix_net_target = MixNet(env.observation_space, recurrent=recurrent)
    mix_net_target.load_state_dict(mix_net.state_dict())
    optimizer = optim.Adam([{'params': q.parameters()}, {'params': mix_net.parameters()}], lr=lr)
    score = 0
    time_steps = 0
    for episode_i in range(max_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * max_episodes)))
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            while not all(done):
                env.render()
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon)
                action = action[0].data.cpu().numpy().tolist()
                next_state, reward, done, info = env.step(action)
                memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))
                score += sum(reward)
                state = next_state
                time_steps += 1

        if memory.size() > warm_up_steps:
            train(q, q_target, mix_net, mix_net_target, memory, optimizer, gamma, batch_size, update_iter, chunk_size)

        if episode_i % update_target_interval == 0:
            q_target.load_state_dict(q.state_dict())
            mix_net_target.load_state_dict(mix_net.state_dict())

        if episode_i % log_interval == 0 and episode_i != 0:
            test_score = evaluate_model(test_env, test_episodes, q)
            train_score = score / log_interval
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, train_score, test_score, memory.size(), epsilon))
            if USE_WANDB:
                wandb.log({'episode': episode_i, 'test-score': test_score,
                           'buffer-size': memory.size(), 'epsilon': epsilon, 'train-score': train_score})
            score = 0

    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    sanitized_env_name = env_name.replace(":", "_")
    q_path = os.path.join(model_dir, f'q_{sanitized_env_name}.pt')
    mix_net_path = os.path.join(model_dir, f'mix_net_{sanitized_env_name}.pt')
    torch.save(q.state_dict(), q_path)
    torch.save(mix_net.state_dict(), mix_net_path)
    print(f"Saved models to {q_path} and {mix_net_path}")

    env.close()
    test_env.close()


def run_from_checkpoint(env_name, q_path, mix_net_path, recurrent, test_episodes=5, record=False):
    env = gym.make(env_name)
    if record:
        if os.path.exists('recordings'):
            shutil.rmtree('recordings')
        env = Monitor(env, directory='recordings', force=True, video_callable=lambda episode_id: True)

    # Init model
    q = QNet(env.observation_space, env.action_space, recurrent)
    q.load_state_dict(torch.load(q_path, map_location=torch.device('cpu')))
    q.eval()

    print(f"Loaded QNet from {q_path}")
    avg_score = test(env, test_episodes, q)
    print(f"Avg test score over {test_episodes} episodes: {avg_score}")

    env.close()


if __name__ == "__main__":
    # Run directly in debugger (no argparse)
    main(
        env_name='ma_gym:Checkers-v0',
        lr=0.001,
        gamma=0.99,
        batch_size=32,
        buffer_limit=50000,
        log_interval=1000,
        max_episodes=100,  # You can reduce this for debugging
        max_epsilon=0.9,
        min_epsilon=0.1,
        test_episodes=5,
        warm_up_steps=2000,
        update_iter=10,
        chunk_size=10,
        update_target_interval=20,
        recurrent=True,
        # recurrent= False,# or False depending on what you wantA recurrent model keeps track of information from previous time steps (or previous inputs) using something called a hidden state.
        record=False        # Set True if you want videos
    )

