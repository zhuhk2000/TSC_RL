from envs import SumoEnv
from agents import MultiDQNAgent
from replay_buffers import ReplayBuffer
import numpy as np
from eval import plot_aql, plot_att, read_trip_file
# from torch.utils.tensorboard import SummaryWriter

env = SumoEnv("./data")
replay_buffers = [ReplayBuffer(10000) for _ in range(len(env.traffic_lights))]

agents = MultiDQNAgent(env, replay_buffers)
train_avg_queue_list = []
eval_avg_queue_list = []
train_avg_travel_time_list = []
eval_avg_travel_time_list = []


# writer = SummaryWriter()
# step = 0
for i in range(1,1000):
    agents.env.reset()
    done = False
    while not done:
        state, reward, done, info = agents.play_step()
        # step += 1
    train_avg_queque = sum(info['queue_count']) / len(info['queue_count'])
    train_avg_queue_list.append(train_avg_queque)
    avg_travel_time = read_trip_file('./data/output/tripinfo.xml')
    train_avg_travel_time_list.append(avg_travel_time)
    # print(f"Episode {i} done")
    # 每10个episode训练结束后进行一次eval
    if i % 10 == 0:
        info = agents.eval()
        avg_queue = sum(info['queue_count']) / len(info['queue_count'])
        eval_avg_queue_list.append(avg_queue)   
        rewards = info['reward']
        cum_reward = np.array(rewards).sum(axis=0)
        avg_travel_time = read_trip_file('./data/output/tripinfo.xml')
        eval_avg_travel_time_list.append(avg_travel_time)
        plot_att(train_avg_travel_time_list, eval_avg_travel_time_list)
        print(f"Episode{i} AQL:{avg_queue:.2f}")
        print(f"Episode{i} ATT:{avg_travel_time:.2f}")
        print(f"Episode{i} Cumulative reward:{cum_reward}")      
        plot_aql(train_avg_queue_list, eval_avg_queue_list)

agents.save_model()




