from envs import SumoEnv
from agents import MultiDQNAgent
from replay_buffers import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from torch.utils.tensorboard import SummaryWriter

env = SumoEnv("./data")
replay_buffers = [ReplayBuffer(10000) for _ in range(len(env.traffic_lights))]

agents = MultiDQNAgent(env, replay_buffers)
train_avg_queue_list = []
eval_avg_queue_list = []
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title('train')
ax2.set_title('eval')
ax1.set_xlabel('Episode')
ax2.set_xlabel('Episode')
ax1.set_ylabel('Average queue length')
ax2.set_ylabel('Average queue length')
# writer = SummaryWriter()
# step = 0
for i in range(500):
    done = False
    agents.env.reset()
    ax1.clear()
    ax2.clear()   
    while not done:
        state, reward, done, info = agents.play_step()
        # step += 1
    test_avg_queque = sum(info['queue_count']) / len(info['queue_count'])
    train_avg_queue_list.append(test_avg_queque)
    print(f"Episode {i} done")
    # 每10个episode训练结束后进行一次eval
    if i % 10 == 0:
        info = agents.eval()
        avg_queue = sum(info['queue_count']) / len(info['queue_count'])
        eval_avg_queue_list.append(avg_queue)   
        rewards = info['reward']
        cum_reward = np.array(rewards).sum(axis=0)
        print(f"Episode{i} AQL:{avg_queue:.2f}")
        print(f"Episode{i} Cumulative reward:{cum_reward}")      
        ax2.plot(eval_avg_queue_list, label='eval')
    ax1.plot(train_avg_queue_list, label='train')
    fig.savefig('train_eval.png')

agents.save_model()




