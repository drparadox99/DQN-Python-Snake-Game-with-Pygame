import os
import torch.optim as optim
from Snake_Env import SnakeEnv
from cls.DQN import DQN
from cls.ReplayBuffer import ReplayBuffer
from cls.DQNTrainer import DQNTrainer

TRAIN = True
FINETUNE = False
SAVE_MODEL = True  # Save the model after training

# Training hyperparameters
TRAINING_EPISODES = 500  # 10 valid only if TRAIN is True
FINETUNE_TRIALS = 100  # valid only if FINETUNE is True

# Set the following hyperparameters if FINETUNE is False
GAMMA = 0.99
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 800
LEARNING_RATE = 1e-3



# Initialize environment, networks, optimizer, and replay buffer
env = SnakeEnv()
state_dim = env.observation_space
action_dim = env.action_space

main_network = DQN(state_dim, action_dim)
target_network = DQN(state_dim, action_dim)
target_network.load_state_dict(main_network.state_dict())
target_network.eval()

replay_buffer = ReplayBuffer(10000)
STEP_COUNT = 0
params = {     'lr': LEARNING_RATE,
				'gamma': GAMMA,
				'batch_size': BATCH_SIZE,
				'target_update_frequency': TARGET_UPDATE_FREQUENCY
			}

#training either with defined hyperparameters either by optimal params obtained by optimizer(optuna)
optimizer = optim.Adam(main_network.parameters(), lr=params['lr'])
trainer = DQNTrainer(env, main_network, target_network, optimizer, replay_buffer,
					 f'{os.path.dirname(__file__)}/model/model.pth', gamma=params['gamma'],
					 batch_size=params['batch_size'],
					 target_update_frequency=params['target_update_frequency'])
trainer.train(TRAINING_EPISODES, save_model=SAVE_MODEL)
