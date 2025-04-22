import torch
class DQN(torch.nn.Module):
	def __init__(self, state_dim, action_dim,out_features=128):
		super(DQN, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, out_features)
		self.fc2 = torch.nn.Linear(out_features, out_features)
		self.fc3 = torch.nn.Linear(out_features, action_dim)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)
