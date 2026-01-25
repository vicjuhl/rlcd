import random
from collections import namedtuple, deque

Transition = namedtuple(
    'Transition',
    ('s', 'a', 'r', 's_next', 'terminal')
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.

    def __len__(self):
        return len(self.memory)