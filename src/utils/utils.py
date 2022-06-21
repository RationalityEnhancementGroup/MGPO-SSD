from collections import deque
from random import sample
import torch
from matplotlib import animation
import matplotlib.pyplot as plt

class Replay_Memory():
    def __init__(self, size=10000):
        self.memory = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        transition = (torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([reward]), torch.Tensor([next_state]), torch.Tensor([done]))
        self.memory.append(transition)

    def sample_batch(self, batch_size = 100):
        batch = sample(self.memory, batch_size)
        state_batch = torch.cat([s1 for (s1,a,r,s2,d) in batch])
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in batch]).long()
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in batch])
        next_state_batch = torch.cat([s2 for (s1,a,r,s2,d) in batch])
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in batch])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)

def save_frames_as_gif(frames, path='./img/', filename='gym_animation.gif'):
    """
    https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

    -- requires imagemagick --
    """
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)