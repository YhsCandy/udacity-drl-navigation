import numpy as np
import random

from model import NoisyDuelingQNetwork, DuelingQNetwork, QNetwork
from buffer import PrioritizedReplayBuffer, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, training, args):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            training (bool): Prepare for training
            args (object): Command line arguments
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        self.seed = seed
        nn_type = args.type.lower()

        self._sample_noise = False
        self._update_buffer_priorities = False

        # NN
        if training:
            self.batch_size = args.batch_size
            self.gamma = args.gamma
            self.tau = args.tau
            self.update_every = args.update_every

            self.qnetwork_local = self._create_nn(nn_type, state_size, action_size, self.seed)
            self.qnetwork_target = self._create_nn(nn_type, state_size, action_size, self.seed)

            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.learning_rate)

            # Replay memory
            self.memory = self._create_buffer(args.buffer.lower(), action_size, args.buffer_size,
                                              self.batch_size, self.seed)
            # Initialize time step (for updating every UPDATE_EVERY steps)
            self.t_step = 0
        else:
            self.qnetwork_local = self._create_nn(nn_type, state_size, action_size, self.seed)

    def _create_buffer(self, buffer_type, action_size, buffer_size, batch_size, seed):
        if buffer_type == 'prioritized':
            self._update_buffer_priorities = True
            return PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed)
        elif buffer_type == 'sample':
            return ReplayBuffer(action_size, buffer_size, batch_size, seed)
        else:
            raise Exception('Unknown buffer type - must be one of prioritized or sample')

    def _create_nn(self, nn_type, state_size, action_size, seed):
        if nn_type == 'noisydueling':
            self._sample_noise = True
            return NoisyDuelingQNetwork(state_size, action_size, seed).to(device)
        elif nn_type == 'dueling':
            return DuelingQNetwork(state_size, action_size, seed).to(device)
        elif nn_type == 'q':
            return QNetwork(state_size, action_size, seed).to(device)
        else:
            raise Exception('Unknown NN type - must be one of NoisyDueling, Dueling or Q')

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            if self._sample_noise:
                self.qnetwork_local.sample_noise()

            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self._update_buffer_priorities:
            states, actions, rewards, next_states, dones, indexes, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        if self._sample_noise:
            self.qnetwork_target.sample_noise()
            self.qnetwork_local.sample_noise()

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Update memory priorities
        if self._update_buffer_priorities:
            self.memory.update_priorities(indexes, (Q_expected - Q_targets).detach().squeeze().abs().cpu().numpy().tolist())

        # Compute loss
        if self._update_buffer_priorities:
            loss = (F.mse_loss(Q_expected, Q_targets) * weights).mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
