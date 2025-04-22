from collections import deque
import random

# Define the experience replay buffer
class ReplayBuffer:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		#state and next state: list of 8 elements
		#action: int
		#reward: float
		#done: float
		#self.buffer.apennd([ (state,action,reward,next_state,donne),(state,action,reward,next_state,donne), ...])
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
		return state, action, reward, next_state, done

	def __len__(self):
		return len(self.buffer)

