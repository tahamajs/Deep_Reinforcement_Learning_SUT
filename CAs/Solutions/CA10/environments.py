import numpy as np
import matplotlib.pyplot as plt


class SimpleGridWorld:
    """Simple gridworld for model learning demonstration"""

    def __init__(self, size=5):
        self.size = size
        self.num_states = size * size
        self.num_actions = 4  # up, down, left, right
        self.reset()

    def reset(self):
        self.state = 0  # Start at top-left
        return self.state

    def step(self, action):
        x, y = self.state % self.size, self.state // self.size

        if action == 0 and y > 0:  # up
            y -= 1
        elif action == 1 and y < self.size - 1:  # down
            y += 1
        elif action == 2 and x > 0:  # left
            x -= 1
        elif action == 3 and x < self.size - 1:  # right
            x += 1

        self.state = y * self.size + x

        if self.state == self.num_states - 1:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False

        return self.state, reward, done


class BlockingMaze:
    """Environment that changes to test Dyna-Q adaptability"""

    def __init__(self, width=9, height=6, change_episode=1000):
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4  # up, down, left, right
        self.change_episode = change_episode
        self.episode_count = 0

        self.start_pos = (0, 3)  # Start position
        self.goal_pos = (8, 0)   # Goal position

        self.blocked_cells = set()
        self.setup_initial_maze()

        self.state = self.pos_to_state(self.start_pos)

    def pos_to_state(self, pos):
        """Convert (x, y) position to state index"""
        return pos[1] * self.width + pos[0]

    def state_to_pos(self, state):
        """Convert state index to (x, y) position"""
        return (state % self.width, state // self.width)

    def setup_initial_maze(self):
        """Setup initial maze with one path blocked"""
        for y in range(1, 4):
            self.blocked_cells.add((3, y))

        self.initial_blocks = self.blocked_cells.copy()

        self.changed_blocks = set()
        for x in range(1, 8):
            self.changed_blocks.add((x, 2))

    def reset(self):
        """Reset environment"""
        self.episode_count += 1

        if self.episode_count == self.change_episode:
            self.blocked_cells = self.changed_blocks.copy()
            print(f"\n*** Environment changed at episode {self.episode_count} ***")

        self.state = self.pos_to_state(self.start_pos)
        return self.state

    def step(self, action):
        """Take action in environment"""
        current_pos = self.state_to_pos(self.state)
        x, y = current_pos

        if action == 0 and y > 0:  # up
            new_pos = (x, y - 1)
        elif action == 1 and y < self.height - 1:  # down
            new_pos = (x, y + 1)
        elif action == 2 and x > 0:  # left
            new_pos = (x - 1, y)
        elif action == 3 and x < self.width - 1:  # right
            new_pos = (x + 1, y)
        else:
            new_pos = current_pos  # Invalid move, stay in place

        if new_pos in self.blocked_cells:
            new_pos = current_pos  # Can't move into blocked cell

        self.state = self.pos_to_state(new_pos)

        if new_pos == self.goal_pos:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return self.state, reward, done

    def render_maze(self):
        """Render current maze state"""
        maze = np.zeros((self.height, self.width))

        for x, y in self.blocked_cells:
            maze[y, x] = -1

        maze[self.start_pos[1], self.start_pos[0]] = 2
        maze[self.goal_pos[1], self.goal_pos[0]] = 3

        current_pos = self.state_to_pos(self.state)
        if current_pos != self.start_pos and current_pos != self.goal_pos:
            maze[current_pos[1], current_pos[0]] = 1

        return maze
