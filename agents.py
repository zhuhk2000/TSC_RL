import os
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn as nn
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter

from model import DQN
from envs import SumoEnv, TrafficLight
from replay_buffers import ReplayBuffer



device = "cuda" if torch.cuda.is_available() else "cpu"

class DQNAgent:
    def __init__(self, tls: TrafficLight, replay_buffer: ReplayBuffer) -> None:
        self.net = DQN(tls.inlane_count, tls.num_phases).to(device)
        self.target_net = DQN(tls.inlane_count, tls.num_phases).to(device)  # target network
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.optimizer = Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = replay_buffer
        self.controlled_tls = tls    
        self.tls_id = tls.tls_id
        self.steps_done = 0
        self.batch_size = 128
        self.target_update = 100
        self.gamma = 0.95


    def select_action(self, state, epsilon: float) -> int:
        self.steps_done += 1
        q_value = self.net(torch.from_numpy(state).unsqueeze(0).to(device))
        if random.random() > epsilon:
            with torch.no_grad():
                return q_value.argmax().item()
        else:
            return random.choice(range(self.controlled_tls.num_phases))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        if self.steps_done % self.target_update == 0:
            self.update_target_network()

        transitions = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)
        state_batch = torch.cat(state_batch)
        next_state_batch = torch.cat(next_state_batch)
        action_batch = torch.cat(action_batch).unsqueeze(-1)
        reward_batch = torch.cat(reward_batch)
        done_batch = torch.cat(done_batch)
        q_value = self.net(state_batch)
        q_value = q_value.gather(1, action_batch).squeeze(-1)
        with torch.no_grad():
            next_q_value = self.target_net(next_state_batch).max(1)[0].detach()
            next_q_value[done_batch] = 0
            next_q_value.detach()
        expected_q_value = reward_batch + self.gamma * next_q_value
        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        import time
        import os
        if not os.path.exists(f"models/{time.strftime('%m-%d-%H-%M')}"):
            os.makedirs(f"models/{time.strftime('%m-%d-%H-%M')}")
        file_name = f"models/{time.strftime('%m-%d-%H-%M')}/{self.tls_id}_model.path"
        torch.save(self.net, file_name)

    def load_model(self, file_path):
        self.net = torch.load(file_path)

    # def save_replay_buffer(self):
    #     pass

    # def load_replay_buffer(self):
    #     pass


class MultiDQNAgent:
    def __init__(self, env: SumoEnv, replay_buffers: List[ReplayBuffer]):
        self.env = env
        self.agents = [DQNAgent(tls, replay_buffer) for tls, replay_buffer in zip(env.traffic_lights, replay_buffers)]
        self.replay_buffers = replay_buffers
        self.epsilon = 0.9
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01


    def train(self):
        for agent in self.agents:
            agent.train()

    def save_model(self):
        for agent in self.agents:
            agent.save_model()

    def load_model(self, dir_path):
        for agent in self.agents:
            file_name = f"{agent.tls_id}_model.path"
            agent.load_model(os.path.join(dir_path, file_name))
    
    def play_step(self):
        states = self.env.get_observation()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        actions = [agent.select_action(state, self.epsilon) for state, agent in zip(states, self.agents)]
        next_states, rewards, done, info = self.env.step(actions)
        for i in range(len(states)):
            self.replay_buffers[i].add(states[i], next_states[i], rewards[i], actions[i], done)
        self.train()

        return next_states, rewards, done, info
    
    def eval(self):
        self.env.reset()
        done = False
        while not done:
            states = self.env.get_observation()
            actions = [agent.select_action(state, epsilon = 0) for state, agent in zip(states, self.agents)]
            next_state, rewards, done, info = self.env.step(actions)
        return info





 