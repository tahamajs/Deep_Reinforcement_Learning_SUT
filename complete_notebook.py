import json
import sys

# Read the notebook
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp24/hws/HW4/SP24_RL_HW4/RL_HW4_Practical.ipynb', 'r') as f:
    notebook = json.load(f)

# Cell 5: Complete Network class
network_code = """class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        ##########################################################
        # Define your network layers
        # 3-layer feedforward network: input -> 128 -> 128 -> output
        ##########################################################
        self.input_layer = torch.nn.Linear(input_dimension, 128)
        self.hidden_layer = torch.nn.Linear(128, 128)
        self.output_layer = torch.nn.Linear(128, output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        ##########################################################
        # Calculate the output
        # Forward pass with ReLU activations
        ##########################################################
        layer_1 = torch.nn.functional.relu(self.input_layer(inpt))
        layer_2 = torch.nn.functional.relu(self.hidden_layer(layer_1))
        output = self.output_activation(self.output_layer(layer_2))

        return output"""

notebook['cells'][5]['source'] = network_code

# Cell 10: Complete SACAgent class - Define critics and actor
sac_agent_init = """class SACAgent:
    def __init__(self, environment, replay_buffer=None, offline=False, learning_rate=3e-4, discount=0.99, buffer_batch_size=100, alpha_init=1, interpolation_factor=0.01):
        self.environment = environment
        self.state_dim = self.environment.observation_space.shape[0]
        self.action_dim = self.environment.action_space.n

        self.alpha_init = alpha_init
        self.learning_rate = learning_rate
        self.discount = discount
        self.buffer_batch_size = buffer_batch_size
        self.interpolation_factor = interpolation_factor

        self.replay_buffer = ReplayBuffer(self.environment) if replay_buffer is None else replay_buffer
        self.offline = offline

        ##########################################################
        # Define critics using your implemented feed forward network
        # Two critic networks for double Q-learning
        # Critics output Q-values for each action
        ##########################################################
        self.critic_local = Network(self.state_dim, self.action_dim)
        self.critic_local2 = Network(self.state_dim, self.action_dim)
        self.critic_optimiser = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate)
        self.critic_optimiser2 = optim.Adam(self.critic_local2.parameters(), lr=self.learning_rate)
        
        # Target networks for stable training
        self.critic_target = Network(self.state_dim, self.action_dim)
        self.critic_target2 = Network(self.state_dim, self.action_dim)
        ##########################################################

        self.soft_update_target_networks(tau=1.)

        ##########################################################
        # Define the actor
        # Actor outputs action probabilities using softmax
        # Define the actor optimizer
        ##########################################################
        self.actor_local = Network(self.state_dim, self.action_dim, output_activation=torch.nn.Softmax(dim=-1))
        self.actor_optimiser = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate)
        ##########################################################
        
        self.target_entropy = 0.98 * -np.log(1 / self.environment.action_space.n)
        self.log_alpha = torch.tensor(np.log(self.alpha_init), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.random.choice(range(self.action_dim), p=action_probabilities)
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.argmax(action_probabilities)
        return discrete_action

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor,
                    next_states_tensor, done_tensor):
        ##########################################################
        # Calculate critic losses
        # 1. Get current Q-values
        # 2. Calculate target Q-values using Bellman equation
        # 3. Compute MSE loss
        ##########################################################
        with torch.no_grad():
            # Get action probabilities and log probs for next states
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            
            # Get Q-values from target networks
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            
            # Take minimum to reduce overestimation bias
            soft_state_values = (action_probabilities * (
                torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)
            
            # Bellman backup
            next_q_values = rewards_tensor + ~done_tensor * self.discount * soft_state_values
        
        # Get current Q-values
        q_values = self.critic_local(states_tensor).gather(1, actions_tensor.long().unsqueeze(-1)).squeeze(-1)
        q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor.long().unsqueeze(-1)).squeeze(-1)
        
        # Compute losses
        critic_loss = F.mse_loss(q_values, next_q_values)
        critic2_loss = F.mse_loss(q_values2, next_q_values)

        return critic_loss, critic2_loss
        ##########################################################

    def actor_loss(self, states_tensor, actions_tensor):
        ##########################################################
        # Implement the actor loss
        # Actor tries to maximize Q-value while maximizing entropy
        ##########################################################
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        
        # Get Q-values from critics
        q_values = self.critic_local(states_tensor)
        q_values2 = self.critic_local2(states_tensor)
        min_q_values = torch.min(q_values, q_values2)
        
        # Actor loss: -E[Q(s,a) - alpha * log(pi(a|s))]
        # Equivalent to: E[alpha * log(pi(a|s)) - Q(s,a)]
        actor_loss = (action_probabilities * (self.alpha * log_action_probabilities - min_q_values)).sum(dim=1).mean()

        return actor_loss, log_action_probabilities
        ##########################################################

    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition)

    def train_networks(self, transition=None, batch_deterministic_start=None):
        ##########################################################
        # Set all the gradients stored in the optimizers to zero
        # Add the new transition to the replay buffer for online case
        ##########################################################
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        
        # Add transition to buffer (only for online learning)
        if not self.offline and transition is not None:
            self.replay_buffer.add_transition(transition)
        
        if self.replay_buffer.get_size() >= self.buffer_batch_size:
            minibatch = self.replay_buffer.sample_minibatch(self.buffer_batch_size, batch_deterministic_start=batch_deterministic_start)
            minibatch_separated = list(map(list, zip(*minibatch)))

            states_tensor = torch.tensor(np.array(minibatch_separated[0]), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]))
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2]), dtype=torch.float32)
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]), dtype=torch.float32)
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))

            ##########################################################
            # Compute the critic loss and perform the backpropagation
            ##########################################################
            critic_loss, critic2_loss = self.critic_loss(states_tensor, actions_tensor, rewards_tensor,
                                                         next_states_tensor, done_tensor)
            critic_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser.zero_grad()
            
            critic2_loss.backward()
            self.critic_optimiser2.step()
            self.critic_optimiser2.zero_grad()
            
            ##########################################################
            # Compute the actor loss and backpropagate the gradient
            ##########################################################
            actor_loss, log_action_probabilities = self.actor_loss(states_tensor, actions_tensor)
            actor_loss.backward()
            self.actor_optimiser.step()
            self.actor_optimiser.zero_grad()
            
            ##########################################################
            # Update alpha (temperature parameter)
            ##########################################################
            alpha_loss = self.temperature_loss(log_action_probabilities)
            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha_optimiser.zero_grad()
            
            self.alpha = self.log_alpha.exp()
            ##########################################################

            self.soft_update_target_networks(self.interpolation_factor)

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state, action):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)"""

notebook['cells'][10]['source'] = sac_agent_init

# Cell 13: Online SAC training loop
online_training_code = """TRAINING_EVALUATION_RATIO = 4
EPISODES_PER_RUN = 500
STEPS_PER_EPISODE = 200
env = gym.make("CartPole-v1")

##########################################################
# Implement the training loop for the online SAC
##########################################################

# Initialize agent
sac_agent = SACAgent(env)

# Training metrics
training_rewards = []
evaluation_rewards = []

print("Starting Online SAC Training...")
for episode in tqdm(range(EPISODES_PER_RUN)):
    state = env.reset()
    episode_reward = 0
    
    for step in range(STEPS_PER_EPISODE):
        # Get action from agent (with exploration)
        action = sac_agent.get_next_action(state, evaluation_episode=False)
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Train the agent
        sac_agent.train_on_transition(state, action, next_state, reward, done)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    training_rewards.append(episode_reward)
    
    # Evaluation every TRAINING_EVALUATION_RATIO episodes
    if (episode + 1) % TRAINING_EVALUATION_RATIO == 0:
        eval_rewards = []
        for _ in range(10):  # 10 evaluation episodes
            eval_state = env.reset()
            eval_episode_reward = 0
            
            for _ in range(STEPS_PER_EPISODE):
                # Get action deterministically for evaluation
                eval_action = sac_agent.get_next_action(eval_state, evaluation_episode=True)
                eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
                
                eval_episode_reward += eval_reward
                eval_state = eval_next_state
                
                if eval_done:
                    break
            
            eval_rewards.append(eval_episode_reward)
        
        mean_eval_reward = np.mean(eval_rewards)
        evaluation_rewards.append(mean_eval_reward)
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/{EPISODES_PER_RUN}, Training Reward: {episode_reward:.2f}, Eval Reward: {mean_eval_reward:.2f}")

env.close()

##########################################################
# Plot the learning curves
##########################################################
plt.figure(figsize=(12, 5))

# Plot training rewards
plt.subplot(1, 2, 1)
plt.plot(training_rewards, alpha=0.6, label='Training Rewards')
plt.plot(np.convolve(training_rewards, np.ones(20)/20, mode='valid'), label='Moving Average (20)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Online SAC - Training Rewards')
plt.legend()
plt.grid(True)

# Plot evaluation rewards
plt.subplot(1, 2, 2)
eval_episodes = list(range(TRAINING_EVALUATION_RATIO, EPISODES_PER_RUN + 1, TRAINING_EVALUATION_RATIO))
plt.plot(eval_episodes, evaluation_rewards, marker='o', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Mean Evaluation Reward')
plt.title('Online SAC - Evaluation Performance')
plt.grid(True)

plt.tight_layout()
plt.savefig('online_sac_learning_curves.png', dpi=150)
plt.show()

print(f"\\nFinal Training Reward: {np.mean(training_rewards[-20:]):.2f}")
print(f"Final Evaluation Reward: {np.mean(evaluation_rewards[-5:]):.2f}")
print(f"\\nReplay Buffer Size: {sac_agent.replay_buffer.get_size()}")"""

notebook['cells'][13]['source'] = online_training_code

# Cell 14: Render online agent
render_online_code = """env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []

state = env.reset()
for _ in range(100):
    frames.append(env.render()[0])
    action = sac_agent.get_next_action(state, evaluation_episode=True)
    state, reward, done, info = env.step(action)
    if done:
        break

env.close()
imageio.mimsave('./online.mp4', frames, fps=25)
show_video('./online.mp4')"""

notebook['cells'][14]['source'] = render_online_code

# Cell 17: Offline SAC training loop
offline_training_code = """NUM_EPOCHS = 200
EPISODES_PER_RUN = 100

env = gym.make("CartPole-v1")

##########################################################
# Implement the training loop for the offline SAC
# Use the replay buffer from the online agent
##########################################################

# Initialize offline agent with the replay buffer from online training
offline_agent = SACAgent(env, replay_buffer=sac_agent.replay_buffer, offline=True)

offline_evaluation_rewards = []

print("Starting Offline SAC Training...")
for epoch in tqdm(range(NUM_EPOCHS)):
    # Train on batches from replay buffer
    for batch_idx in range(100):  # 100 training steps per epoch
        offline_agent.train_networks(transition=None, batch_deterministic_start=None)
    
    # Evaluation
    eval_rewards = []
    for _ in range(EPISODES_PER_RUN):
        eval_state = env.reset()
        eval_episode_reward = 0
        
        for _ in range(200):
            eval_action = offline_agent.get_next_action(eval_state, evaluation_episode=True)
            eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
            
            eval_episode_reward += eval_reward
            eval_state = eval_next_state
            
            if eval_done:
                break
        
        eval_rewards.append(eval_episode_reward)
    
    mean_eval_reward = np.mean(eval_rewards)
    offline_evaluation_rewards.append(mean_eval_reward)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Eval Reward: {mean_eval_reward:.2f}")

env.close()

##########################################################
# Plot the learning curves
##########################################################
plt.figure(figsize=(10, 6))
plt.plot(offline_evaluation_rewards, linewidth=2, label='Offline SAC')
plt.xlabel('Epoch')
plt.ylabel('Mean Evaluation Reward')
plt.title('Offline SAC - Evaluation Performance')
plt.legend()
plt.grid(True)
plt.savefig('offline_sac_learning_curves.png', dpi=150)
plt.show()

print(f"\\nFinal Offline Evaluation Reward: {np.mean(offline_evaluation_rewards[-10:]):.2f}")"""

notebook['cells'][17]['source'] = offline_training_code

# Cell 18: Render offline agent
render_offline_code = """env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []

state = env.reset()
for _ in range(100):
    frames.append(env.render()[0])
    action = offline_agent.get_next_action(state, evaluation_episode=True)
    state, reward, done, info = env.step(action)
    if done:
        break
env.close()
imageio.mimsave('./offline.mp4', frames, fps=25)
show_video('./offline.mp4')"""

notebook['cells'][18]['source'] = render_offline_code

# Cell 21: Collect expert data
collect_expert_code = """env = gym.make('CartPole-v1')
num_episodes = 1000
expert_data = []

##########################################################
# Collect state-action pairs using the trained online agent
##########################################################
print("Collecting expert data...")
for episode in tqdm(range(num_episodes)):
    state = env.reset()
    done = False
    
    while not done:
        # Get action from trained online agent (deterministic)
        action = sac_agent.get_next_action(state, evaluation_episode=True)
        
        # Store state-action pair
        expert_data.append((state, action))
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        state = next_state

env.close()

print(f"Collected {len(expert_data)} state-action pairs from {num_episodes} episodes")"""

notebook['cells'][21]['source'] = collect_expert_code

# Cell 23: BC Model
bc_model_code = """class BCModel(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(BCModel, self).__init__()
        ##########################################################
        # Define the model
        # 2-layer feedforward network for behavioral cloning
        ##########################################################
        self.fc1 = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, x):
        ##########################################################
        # Perform forward pass
        ##########################################################
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)  # Logits for classification
        return output"""

notebook['cells'][23]['source'] = bc_model_code

# Cell 25: BC training loop
bc_training_code = """# Initialize BC model
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

bc_model = BCModel(input_dimension=state_dim, hidden_dimension=128, output_dimension=action_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bc_model.parameters(), lr=1e-3)

num_epochs = 5
batch_size = 32

##########################################################
# Implement behavioral cloning training loop
##########################################################
print("Training Behavioral Cloning model...")

# Prepare data
states = torch.tensor([data[0] for data in expert_data], dtype=torch.float32)
actions = torch.tensor([data[1] for data in expert_data], dtype=torch.long)

# Create dataset
dataset = torch.utils.data.TensorDataset(states, actions)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_states, batch_actions in tqdm(dataloader):
        # Forward pass
        predictions = bc_model(batch_states)
        
        # Compute loss
        loss = criterion(predictions, batch_actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}'"""

notebook['cells'][25]['source'] = bc_training_code

# Cell 27: Render BC agent
render_bc_code = """env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []

state = env.reset()
for _ in range(100):
    frames.append(env.render()[0])
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # Get action from BC model
    with torch.no_grad():
        action_logits = bc_model(state_tensor)
        action = torch.argmax(action_logits, dim=1).item()
    
    state, reward, done, info = env.step(action)
    if done:
        break
        
env.close()
imageio.mimsave('./bc.mp4', frames, fps=25)
show_video('./bc.mp4')"""

notebook['cells'][27]['source'] = render_bc_code

# Write the updated notebook
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp24/hws/HW4/SP24_RL_HW4/RL_HW4_Practical.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook updated successfully!")
