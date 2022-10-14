import numpy as np
import random
from collections import namedtuple

class ChainBuffer:
    """A class for saving information about reconstructed position, pe, ke, log_px, log_pvel
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = int(0)
        self.chain_indices = []
        self.transition = namedtuple("Transition", ("x", "pe", "ke", "log_px", "log_pvel"))

    def update_chain_position(self):
        self.chain_indices.append(self.position - 1)
    
    def push(self, x, pe, ke, log_px, log_pvel):
        to_add = [x, pe, ke, log_px, log_pvel]
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        else:
            raise ValueError("Increase Buffer Capacity")
        self.buffer[self.position] = self.transition(*to_add)
        self.position = int((self.position + 1) % self.capacity)
        self.update_chain_position()
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, prefix=""):
        transitions = self.buffer
        to_save = self.transition(*zip(*transitions))
        np.save(prefix + "x.npy", to_save.x)
        np.save(prefix + "pe.npy", to_save.pe)
        np.save(prefix + "ke.npy", to_save.ke)
        np.save(prefix + "log_px.npy", to_save.log_px)
        np.save(prefix + "log_pvel.npy", to_save.log_pvel)
        np.save(prefix + "chain_indices.npy", self.chain_indices)
