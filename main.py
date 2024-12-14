from envs import SumoEnv
from agents import MultiDQNAgent
from replay_buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

env = SumoEnv("./data")
replay_buffers = [ReplayBuffer(10000) for _ in range(len(env.traffic_lights))]

agents = MultiDQNAgent(env, replay_buffers)
writer = SummaryWriter()
step = 0
for i in range(1000):
    done = False
    agents.env.reset()
    
    while not done:
        state, reward, done, info = agents.play_step()
        step += 1

    print(f"Episode {i} done")
    # 每次训练结束后进行一次eval



