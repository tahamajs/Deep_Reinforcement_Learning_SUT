import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt


class SimpleGridWorld:
    """Simple grid world environment for RL experiments"""
    
    def __init__(self, size=5, goal_reward=10.0, step_penalty=-0.1, max_steps=100):
        self.size = size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.num_actions = 4
        self.actions = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Random start position
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        
        # Random goal position (different from start)
        self.goal_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        
        self.step_count = 0
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        # Return normalized position coordinates
        return np.array([
            self.agent_pos[0] / self.size,
            self.agent_pos[1] / self.size
        ], dtype=np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
        
        # Move agent
        dx, dy = self.actions[action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        
        self.agent_pos = [new_x, new_y]
        self.step_count += 1
        
        # Check if goal reached
        done = (self.agent_pos == self.goal_pos) or (self.step_count >= self.max_steps)
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
        else:
            # Distance-based reward
            distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = self.step_penalty - 0.1 * distance
        
        return self.get_state(), reward, done, {'step_count': self.step_count}
    
    def render(self, mode='rgb_array'):
        """Render the environment"""
        grid = np.zeros((self.size, self.size, 3))
        
        # Goal (green)
        grid[self.goal_pos[0], self.goal_pos[1]] = [0, 1, 0]
        
        # Agent (red)
        grid[self.agent_pos[0], self.agent_pos[1]] = [1, 0, 0]
        
        if mode == 'rgb_array':
            return grid
        elif mode == 'human':
            plt.figure(figsize=(6, 6))
            plt.imshow(grid, interpolation='nearest')
            plt.title('Grid World')
            plt.xticks(range(self.size))
            plt.yticks(range(self.size))
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def get_action_meanings(self):
        """Return action meanings"""
        return ['UP', 'RIGHT', 'DOWN', 'LEFT']


class MultiAgentGridWorld:
    """Multi-agent grid world for MARL experiments"""
    
    def __init__(self, size=5, n_agents=2, goal_reward=10.0, collision_penalty=-1.0, max_steps=100):
        self.size = size
        self.n_agents = n_agents
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.max_steps = max_steps
        
        # Action space: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.num_actions = 5
        self.actions = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
            4: (0, 0)    # stay
        }
        
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.agent_positions = []
        self.goal_positions = []
        
        # Place agents and goals randomly
        for i in range(self.n_agents):
            # Agent position
            pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            while pos in self.agent_positions:
                pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            self.agent_positions.append(pos)
            
            # Goal position
            goal = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            while goal in self.agent_positions or goal in self.goal_positions:
                goal = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            self.goal_positions.append(goal)
        
        self.step_count = 0
        return self.get_observations()
    
    def get_observations(self):
        """Get observations for all agents"""
        observations = []
        for i in range(self.n_agents):
            # Each agent observes: own position, own goal, other agents' positions
            obs = []
            obs.extend([self.agent_positions[i][0] / self.size, self.agent_positions[i][1] / self.size])
            obs.extend([self.goal_positions[i][0] / self.size, self.goal_positions[i][1] / self.size])
            
            # Other agents' positions
            for j in range(self.n_agents):
                if i != j:
                    obs.extend([self.agent_positions[j][0] / self.size, self.agent_positions[j][1] / self.size])
            
            observations.append(np.array(obs, dtype=np.float32))
        return observations
    
    def step(self, actions):
        """Execute actions for all agents"""
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")
        
        rewards = []
        old_positions = [pos.copy() for pos in self.agent_positions]
        
        # Move agents
        for i, action in enumerate(actions):
            if action not in self.actions:
                raise ValueError(f"Invalid action {action} for agent {i}")
            
            dx, dy = self.actions[action]
            new_x = max(0, min(self.size - 1, self.agent_positions[i][0] + dx))
            new_y = max(0, min(self.size - 1, self.agent_positions[i][1] + dy))
            
            # Check for collisions with other agents
            new_pos = [new_x, new_y]
            collision = False
            for j in range(self.n_agents):
                if i != j and new_pos == self.agent_positions[j]:
                    collision = True
                    break
            
            if not collision:
                self.agent_positions[i] = new_pos
        
        self.step_count += 1
        
        # Calculate rewards
        for i in range(self.n_agents):
            if self.agent_positions[i] == self.goal_positions[i]:
                reward = self.goal_reward
            else:
                # Distance-based reward
                distance = abs(self.agent_positions[i][0] - self.goal_positions[i][0]) + \
                          abs(self.agent_positions[i][1] - self.goal_positions[i][1])
                reward = -0.1 - 0.1 * distance
            
            # Check for collisions
            for j in range(self.n_agents):
                if i != j and self.agent_positions[i] == self.agent_positions[j]:
                    reward += self.collision_penalty
            
            rewards.append(reward)
        
        # Check if done
        done = (all(self.agent_positions[i] == self.goal_positions[i] for i in range(self.n_agents)) or 
                self.step_count >= self.max_steps)
        
        return self.get_observations(), rewards, done, {'step_count': self.step_count}
    
    def render(self, mode='rgb_array'):
        """Render the multi-agent environment"""
        grid = np.zeros((self.size, self.size, 3))
        
        # Goals (green)
        for goal in self.goal_positions:
            grid[goal[0], goal[1]] = [0, 1, 0]
        
        # Agents (different colors)
        colors = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        for i, pos in enumerate(self.agent_positions):
            color = colors[i % len(colors)]
            grid[pos[0], pos[1]] = color
        
        if mode == 'rgb_array':
            return grid
        elif mode == 'human':
            plt.figure(figsize=(8, 8))
            plt.imshow(grid, interpolation='nearest')
            plt.title('Multi-Agent Grid World')
            plt.xticks(range(self.size))
            plt.yticks(range(self.size))
            plt.grid(True, alpha=0.3)
            plt.show()


class StochasticGridWorld(SimpleGridWorld):
    """Grid world with stochastic transitions"""
    
    def __init__(self, size=5, goal_reward=10.0, step_penalty=-0.1, max_steps=100, 
                 wind_prob=0.1, wind_strength=1):
        super().__init__(size, goal_reward, step_penalty, max_steps)
        self.wind_prob = wind_prob
        self.wind_strength = wind_strength

    def step(self, action):
        """Execute action with stochastic wind effect"""
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
        
        # Apply wind with probability
        if np.random.random() < self.wind_prob:
            # Random wind direction
            wind_action = np.random.randint(0, 4)
            dx, dy = self.actions[wind_action]
        else:
            dx, dy = self.actions[action]
        
        # Move agent
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        
        self.agent_pos = [new_x, new_y]
        self.step_count += 1
        
        # Check if goal reached
        done = (self.agent_pos == self.goal_pos) or (self.step_count >= self.max_steps)
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
        else:
            # Distance-based reward with wind penalty
            distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            wind_penalty = -0.5 if np.random.random() < self.wind_prob else 0
            reward = self.step_penalty - 0.1 * distance + wind_penalty
        
        return self.get_state(), reward, done, {'step_count': self.step_count, 'wind_applied': np.random.random() < self.wind_prob}


class PartiallyObservableGridWorld(SimpleGridWorld):
    """Grid world with partial observability"""
    
    def __init__(self, size=5, goal_reward=10.0, step_penalty=-0.1, max_steps=100, 
                 observation_radius=2):
        super().__init__(size, goal_reward, step_penalty, max_steps)
        self.observation_radius = observation_radius
        
    def get_state(self):
        """Get partial observation around agent"""
        obs_size = 2 * self.observation_radius + 1
        observation = np.zeros((obs_size, obs_size, 3))
        
        for i in range(obs_size):
            for j in range(obs_size):
                world_x = self.agent_pos[0] + i - self.observation_radius
                world_y = self.agent_pos[1] + j - self.observation_radius
                
                if 0 <= world_x < self.size and 0 <= world_y < self.size:
                    if [world_x, world_y] == self.goal_pos:
                        observation[i, j] = [0, 1, 0]  # Goal
                    else:
                        observation[i, j] = [0.5, 0.5, 0.5]  # Empty space
                else:
                    observation[i, j] = [0, 0, 0]  # Wall
        
        # Agent is at center
        observation[self.observation_radius, self.observation_radius] = [1, 0, 0]
        
        return observation.flatten().astype(np.float32)