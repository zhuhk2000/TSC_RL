from collections import namedtuple, deque
import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

device = "cuda" if torch.cuda.is_available() else "cpu"



class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
    ):
        self._capacity = capacity
        self._num_added = 0
        self._storage = deque(maxlen=capacity)

    def add(self, state, next_state, reward, action, done) -> None:
        if reward is not None:
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
            action = torch.tensor(action).unsqueeze(0).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)
            done = torch.tensor(done, dtype=torch.bool).unsqueeze(0).to(device)
            transition = Transition(state, action, next_state, reward, done)
            self._storage.append(transition)
            self._num_added += 1

    def sample(self, batch_size: int = 1):
        indices = np.random.choice(len(self._storage), batch_size, replace=False)
        samples = [self._storage[idx] for idx in indices]
        return samples
    
    def __len__(self):
        return len(self._storage)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return min(self._num_added, self._capacity)

    @property
    def steps_done(self) -> int:
        return self._num_added