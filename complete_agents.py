#!/usr/bin/env python3
"""
Complete PPO and DDPG Agent implementations for the notebook
"""

ppo_agent_code = '''class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)        
    """

    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        gamma: float,
        tau: float,
        epsilon: float,
        epoch: int,
        rollout_len: int,
        entropy_weight: float,
    ):
        """Initialize."""
        # ==================================== Your Code (Begin) ====================================
        # 1. set hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.entropy_weight = entropy_weight
        
        # 2. check device: cpu/GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 3. init actor critic networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.actor = PPOActor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        
        # 4. set Optimizer for each network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # 5. consider memory for training
        self.states: List = []
        self.actions: List = []
        self.rewards: List = []
        self.values: List = []
        self.log_probs: List = []
        self.dones: List = []
        
        # 6. set total step counts equal to 1
        self.total_step = 1
        
        # 7. define a mode for train/test
        self.is_test = False
        # ==================================== Your Code (End) ====================================

    
        

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # ==================================== Your Code (Begin) ====================================
        # 1. select action for train or test mode
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            mu, std = self.actor(state)
            value = self.critic(state)
        
        if self.is_test:
            # Use mean action during testing
            action = mu
        else:
            # Sample from policy during training
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # 2. if you are in train mode take care of filing considered memory
            self.states.append(state.cpu().numpy())
            self.actions.append(action.cpu().numpy())
            self.values.append(value.cpu().numpy())
            self.log_probs.append(log_prob.cpu().numpy())
        
        return action.cpu().numpy()
        # ==================================== Your Code (End) ====================================


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        # ==================================== Your Code (Begin) ====================================
        next_state, reward, done, _ = self.env.step(action)
        
        if not self.is_test:
            self.rewards.append(reward)
            self.dones.append(done)
        
        return next_state, reward, done
        # ==================================== Your Code (End) ====================================


    def update_model(
        self, next_state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        # ==================================== Your Code (Begin) ====================================
        # 1. set device
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(-1).to(self.device)
        
        # Compute returns and advantages using GAE
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state_tensor).cpu().detach().numpy()
        
        returns = []
        advantages = []
        gae = 0
        
        # 2. for each step:
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value * (1 - dones[i])
            else:
                next_val = values[i + 1] * (1 - dones[i])
            
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.tau * (1 - dones[i]) * gae
            
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(-1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        actor_losses = []
        critic_losses = []
        
        for state, action, old_value, old_log_prob, return_, advantage in ppo_iter(
            self.epoch,
            self.batch_size,
            states,
            actions,
            values,
            log_probs,
            returns,
            advantages,
        ):
            # 3. calculate ratios
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            ratio = torch.exp(log_prob - old_log_prob)
            
            # 4. calculate actor_loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 5. calculate entropy
            entropy = dist.entropy().sum(dim=-1).mean()
            actor_loss = actor_loss - self.entropy_weight * entropy
            
            # 6. calculate critic_loss
            value = self.critic(state)
            critic_loss = F.mse_loss(value, return_)
            
            # 7. Train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 8. Train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # Clear memory
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
        
        return np.mean(actor_losses), np.mean(critic_losses)
        # ==================================== Your Code (End) ====================================

    

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        # ==================================== Your Code (Begin) ====================================
        # 1. set the status of trainig
        self.is_test = False
        
        # 2. Reset environment
        state = self.env.reset()
        actor_losses, critic_losses, scores = [], [], []
        score = 0
        
        # 3. for number of frames:
        for frame_idx in range(1, num_frames + 1):
            # 4. select an action
            action = self.select_action(state)
            
            # 5. step in environment
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
            self.total_step += 1
            
            # 6. update model
            if self.total_step % self.rollout_len == 0:
                actor_loss, critic_loss = self.update_model(next_state)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # Episode ended
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
            
            # Plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, actor_losses, critic_losses)
        
        # 7. terminate environment after training is finished
        self.env.close()
        # ==================================== Your Code (End) ====================================

        

    def test(self):
        """Test the agent."""
        # ==================================== Your Code (Begin) ====================================
        # 1. set the status of trainig
        self.is_test = True
        
        # 2. Reset environment
        state = self.env.reset()
        done = False
        score = 0
        frames = []
        
        # 3. roll out one episode living in the environment and save frames for getting render
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
        
        print(f"Test Score: {score:.2f}")
        self.env.close()
        
        return frames
        # ==================================== Your Code (End) ====================================


    def _plot(
        self,
        frame_idx: int,
        scores: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
    ):
        """Plot the training progresses."""
        # ==================================== Your Code (Begin) ====================================
        # 1. define a function for sub plots
        clear_output(True)
        plt.figure(figsize=(20, 5))
        
        # 2. for each variable plot the specific subplot
        plt.subplot(131)
        plt.title(f"Frame {frame_idx}. Score: {np.mean(scores[-10:]) if scores else 0:.2f}")
        plt.plot(scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        
        plt.subplot(132)
        plt.title(f"Actor Loss")
        plt.plot(actor_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        plt.subplot(133)
        plt.title(f"Critic Loss")
        plt.plot(critic_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        # 3. plot the whole figure
        plt.show()
        # ==================================== Your Code (End) ====================================
'''

ddpg_agent_code = '''class DDPGAgent:
    """DDPGAgent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 10000,
    ):
        """Initialize."""

        # ==================================== Your Code (Begin) ====================================
        # 1. initialize hyper parameters, reply buffer and environment
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.initial_random_steps = int(initial_random_steps)
        
        # 2. set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # 4. init actor network
        self.actor = DDPGActor(obs_dim, action_dim).to(self.device)
        self.actor_target = DDPGActor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # 5. init value fuction (value critic)
        self.critic = DDPGCritic(obs_dim + action_dim).to(self.device)
        self.critic_target = DDPGCritic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 7. set Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Replay buffer
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        
        # 6. init OUNoise
        self.noise = OUNoise(action_dim, theta=ou_noise_theta, sigma=ou_noise_sigma)
        
        # consider stroring transitions in memeory, counting steps and specify train/test mode
        self.transition = []
        self.total_step = 0
        self.is_test = False
        # ==================================== Your Code (End) ====================================
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        
        # ==================================== Your Code (Begin) ====================================
        # 1. check if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                selected_action = self.actor(state_tensor).cpu().numpy()
        
        # 2. add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        # 3. store transition
        self.transition = [state, selected_action]
        
        # return selected action
        return selected_action
        # ==================================== Your Code (End) ====================================
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        # ==================================== Your Code (Begin) ====================================
        # step in environment and save transition in memory if you are not in test mode
        next_state, reward, done, _ = self.env.step(action)
        
        if not self.is_test:
            self.transition.extend([reward, next_state, done])
            self.memory.store(*self.transition)
        
        return next_state, reward, done
        # ==================================== Your Code (End) ====================================
    
    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        # ==================================== Your Code (Begin) ====================================
        # 1. set device
        # 2. get a batch from memory and calculate the return
        samples = self.memory.sample_batch()
        states = torch.FloatTensor(samples["obs"]).to(self.device)
        actions = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        next_states = torch.FloatTensor(samples["next_obs"]).to(self.device)
        dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        
        # 3. calculate the loss for actor and critic networks
        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Critic loss
        q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss (policy gradient)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 4. update target
        self._target_soft_update()
        
        return actor_loss.item(), critic_loss.item()
        # ==================================== Your Code (End) ====================================
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        # ==================================== Your Code (Begin) ====================================
        # 1. set the status of trainig
        self.is_test = False
        
        # 2. Reset environment
        state = self.env.reset()
        actor_losses, critic_losses, scores = [], [], []
        score = 0
        
        # 3. for number of frames:
        for frame_idx in range(1, num_frames + 1):
            # 4. select an action
            action = self.select_action(state)
            
            # 5. step in environment
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
            self.total_step += 1
            
            # 6. update model
            if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # Episode ended
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
                self.noise.reset()
            
            # 7. plot the computed variables
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, actor_losses, critic_losses)
        
        # 8. terminate environment after training is finished
        self.env.close()
        # ==================================== Your Code (End) ====================================
        
    def test(self):
        """Test the agent."""
        # ==================================== Your Code (Begin) ====================================
        # 1. set the status of trainig
        self.is_test = True
        
        # 2. Reset environment
        state = self.env.reset()
        done = False
        score = 0
        frames = []
        
        # 3. roll out one episode living in the environment and save frames for getting render
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
        
        print(f"Test Score: {score:.2f}")
        self.env.close()
        
        return frames
        # ==================================== Your Code (End) ====================================
    
    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        # ==================================== Your Code (Begin) ====================================
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # ==================================== Your Code (End) ====================================
    
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        actor_losses: List[float], 
        critic_losses: List[float], 
    ):
        """Plot the training progresses."""
        # ==================================== Your Code (Begin) ====================================
        clear_output(True)
        plt.figure(figsize=(20, 5))
        
        plt.subplot(131)
        plt.title(f"Frame {frame_idx}. Score: {np.mean(scores[-10:]) if scores else 0:.2f}")
        plt.plot(scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        
        plt.subplot(132)
        plt.title(f"Actor Loss")
        plt.plot(actor_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        plt.subplot(133)
        plt.title(f"Critic Loss")
        plt.plot(critic_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        plt.show()
        # ==================================== Your Code (End) ====================================
'''

print("PPO Agent Code:")
print("=" * 80)
print(ppo_agent_code)
print("\n\n")
print("DDPG Agent Code:")
print("=" * 80)
print(ddpg_agent_code)


