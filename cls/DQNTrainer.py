import torch
import os

class DQNTrainer:
	def __init__(self, env, main_network, target_network, optimizer, replay_buffer, model_path='model/model.pth',
				 gamma=0.99, batch_size=64, target_update_frequency=50):
		self.env = env
		self.main_network = main_network
		self.target_network = target_network
		self.optimizer = optimizer
		self.replay_buffer = replay_buffer
		self.model_path = model_path
		self.gamma = gamma
		self.batch_size = batch_size
		self.target_update_frequency = target_update_frequency
		self.step_count = 0

		# Load the model if it exists
		if os.path.exists(os.path.dirname(self.model_path)):
			if os.path.isfile(self.model_path):
				self.main_network.load_state_dict(torch.load(self.model_path))
				self.target_network.load_state_dict(torch.load(self.model_path))
				print("Loaded model from disk")
		else:
			os.makedirs(os.path.dirname(self.model_path))

	def train(self, num_episodes, save_model):
		total_rewards = []
		for episode in range(num_episodes):
			state = self.env.reset()  # Extract the state from the returned tuple
			done = False
			total_reward = 0
			while not done:
				# render environment
				self.env.render()
				#select action
				action = self.main_network(torch.FloatTensor(state).unsqueeze(0)).argmax(dim=1).item() #index of the max value along  columns (in a row)
				#take action in env, render the env and return new_stae, retard and if episode is done
				next_state, reward, done = self.env.step(action)  # Extract the next_state from the returned tuple
				#add to replar buffer (grows at each episode)
				self.replay_buffer.push(state, action, reward, next_state, done)
				state = next_state
				total_reward += reward

				#update main network and occasionally the target network
				if len(self.replay_buffer) >= self.batch_size:
					self.update_network()

			total_rewards.append(total_reward)
			print(f"Episode {episode}, Total Reward: {total_reward}")

		# Save the model after training
		if save_model:
			torch.save(self.main_network.state_dict(), self.model_path)
			print("Saved model to disk")

		self.env.close()
		return sum(total_rewards) / len(total_rewards)  # Return average reward

	def update_network(self):
		state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(
			self.batch_size)

		# Convert to tensors
		state_batch = torch.FloatTensor(state_batch)
		action_batch = torch.LongTensor(action_batch)
		reward_batch = torch.FloatTensor(reward_batch)
		next_state_batch = torch.FloatTensor(next_state_batch)
		done_batch = torch.FloatTensor(done_batch)

		# Calculate the current Q-values
		#gather(1,...): dim=1 means we want to select across columns, i.e., select an action per row.
		q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

		# Calculate the target Q-values
		#max return(max_values,indices)
		next_q_values = self.target_network(next_state_batch).max(1)[0]

		#done: 1 pour fin d'épisode 0 sinon. if done = 1 alros expected_q_values = reward_batch
		expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

		# Compute the loss
		loss = torch.nn.MSELoss()(q_values, expected_q_values.detach())

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Periodically update the target network
		if self.step_count % self.target_update_frequency == 0:
			self.target_network.load_state_dict(self.main_network.state_dict())
		self.step_count += 1
