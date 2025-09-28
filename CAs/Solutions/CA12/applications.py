# Comprehensive Multi-Agent Applications
import torch


class ResourceAllocationEnvironment:
    """Environment for resource allocation problems."""

    def __init__(self, n_agents=4, n_resources=10, resource_capacity=5):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.resource_capacity = resource_capacity

        # Resource demands and values
        self.resource_demands = torch.randint(
            1, resource_capacity + 1, (n_agents, n_resources)
        )
        self.resource_values = torch.rand(n_agents, n_resources) * 10

        # Current allocation
        self.current_allocation = torch.zeros(n_agents, n_resources, dtype=torch.int)

        # Resource constraints
        self.resource_limits = torch.full(
            (n_resources,), resource_capacity, dtype=torch.int
        )

    def reset(self):
        """Reset environment."""
        self.current_allocation = torch.zeros(
            self.n_agents, self.n_resources, dtype=torch.int
        )
        return self.get_state()

    def get_state(self):
        """Get current state representation."""
        return {
            "allocation": self.current_allocation.clone(),
            "remaining_capacity": self.resource_limits
            - self.current_allocation.sum(dim=0),
            "agent_demands": self.resource_demands.clone(),
            "agent_values": self.resource_values.clone(),
        }

    def step(self, agent_actions):
        """
        Execute agent actions.

        Args:
            agent_actions: [n_agents, n_resources] - allocation requests
        """
        # Validate actions - clamp each agent's actions to their demands
        agent_actions = torch.min(agent_actions, self.resource_demands)

        # Check resource constraints
        total_requested = agent_actions.sum(dim=0)
        available_capacity = self.resource_limits - self.current_allocation.sum(dim=0)

        # Allocate resources proportionally if over-subscribed
        allocation_ratios = torch.where(
            total_requested > available_capacity,
            available_capacity.float() / total_requested.float(),
            torch.ones_like(total_requested.float()),
        )

        # Apply allocations
        actual_allocation = (agent_actions.float() * allocation_ratios).int()
        self.current_allocation += actual_allocation

        # Compute rewards
        rewards = []
        for i in range(self.n_agents):
            agent_allocation = actual_allocation[i]
            agent_value = (agent_allocation.float() * self.resource_values[i]).sum()
            agent_satisfaction = (
                agent_allocation.float() / self.resource_demands[i].float()
            )
            reward = (
                agent_value + agent_satisfaction.mean() * 5
            )  # Bonus for satisfaction
            rewards.append(reward)

        # Check if episode is done
        done = (self.current_allocation.sum(dim=0) >= self.resource_limits * 0.9).all()

        next_state = self.get_state()

        return next_state, torch.tensor(rewards), done, {}

    def render(self):
        """Render environment state."""
        print("Resource Allocation State:")
        print(f"Current allocation:\n{self.current_allocation}")
        print(
            f"Remaining capacity: {self.resource_limits - self.current_allocation.sum(dim=0)}"
        )
        print(
            f"Total utilization: {(self.current_allocation.sum() / self.resource_limits.sum() * 100):.1f}%"
        )


class AutonomousVehicleEnvironment:
    """Multi-agent autonomous vehicle coordination environment."""

    def __init__(self, n_vehicles=5, road_length=100, speed_limit=30):
        self.n_vehicles = n_vehicles
        self.road_length = road_length
        self.speed_limit = speed_limit

        # Vehicle states: [position, speed, lane]
        self.vehicle_states = torch.zeros(n_vehicles, 3)
        self.vehicle_states[:, 0] = torch.linspace(
            0, road_length * 0.8, n_vehicles
        )  # Initial positions
        self.vehicle_states[:, 1] = (
            torch.rand(n_vehicles) * speed_limit * 0.5
        )  # Initial speeds
        self.vehicle_states[:, 2] = torch.randint(
            0, 3, (n_vehicles,)
        )  # Lanes (0, 1, 2)

        # Road network
        self.n_lanes = 3
        self.lane_speeds = torch.tensor([25, 30, 35])  # Different speed limits per lane

        # Safety constraints
        self.min_safe_distance = 10.0
        self.collision_penalty = -100

    def reset(self):
        """Reset environment."""
        self.vehicle_states[:, 0] = torch.linspace(
            0, self.road_length * 0.8, self.n_vehicles
        )
        self.vehicle_states[:, 1] = torch.rand(self.n_vehicles) * self.speed_limit * 0.5
        self.vehicle_states[:, 2] = torch.randint(0, self.n_lanes, (self.n_vehicles,))
        return self.get_state()

    def get_state(self):
        """Get current state for all vehicles."""
        states = []
        for i in range(self.n_vehicles):
            # Local observation: own state + nearby vehicles
            own_state = self.vehicle_states[i]

            # Find vehicles in front and behind
            same_lane = self.vehicle_states[:, 2] == own_state[2]
            positions = self.vehicle_states[:, 0]

            # Vehicle in front
            front_mask = (positions > own_state[0]) & same_lane
            if front_mask.any():
                front_vehicle_idx = torch.argmin(positions[front_mask])
                front_vehicle_pos = positions[front_mask][front_vehicle_idx]
                front_distance = front_vehicle_pos - own_state[0]
            else:
                front_distance = self.road_length - own_state[0]

            # Vehicle behind
            back_mask = (positions < own_state[0]) & same_lane
            if back_mask.any():
                back_vehicle_idx = torch.argmax(positions[back_mask])
                back_vehicle_pos = positions[back_mask][back_vehicle_idx]
                back_distance = own_state[0] - back_vehicle_pos
            else:
                back_distance = own_state[0]

            # Adjacent lane vehicles
            left_lane = (own_state[2] - 1).clamp(0, self.n_lanes - 1)
            right_lane = (own_state[2] + 1).clamp(0, self.n_lanes - 1)

            state = torch.tensor(
                [
                    own_state[0] / self.road_length,  # Normalized position
                    own_state[1] / self.speed_limit,  # Normalized speed
                    own_state[2] / (self.n_lanes - 1),  # Normalized lane
                    front_distance / 50.0,  # Normalized front distance
                    back_distance / 50.0,  # Normalized back distance
                    (self.lane_speeds[int(own_state[2])] - own_state[1])
                    / self.speed_limit,  # Speed difference from lane limit
                ]
            )

            states.append(state)

        return torch.stack(states)

    def step(self, actions):
        """
        Execute vehicle actions.

        Actions: [acceleration, lane_change] for each vehicle
        """
        rewards = []
        done = False

        for i in range(self.n_vehicles):
            action = actions[i]
            state = self.vehicle_states[i]

            # Parse action: [accel_command, lane_change_command]
            accel_command = action[0] * 5 - 2.5  # Convert to [-2.5, 2.5] m/s²
            lane_change = action[1] * 2 - 1  # Convert to [-1, 1]

            # Update speed
            new_speed = (state[1] + accel_command).clamp(0, self.speed_limit)

            # Update lane (discrete change)
            if abs(lane_change) > 0.5:
                new_lane = (state[2] + (1 if lane_change > 0 else -1)).clamp(
                    0, self.n_lanes - 1
                )
            else:
                new_lane = state[2]

            # Update position
            new_position = (state[0] + new_speed).clamp(0, self.road_length)

            # Update state
            self.vehicle_states[i] = torch.tensor([new_position, new_speed, new_lane])

            # Compute reward
            reward = 0

            # Speed reward (prefer lane-appropriate speed)
            target_speed = self.lane_speeds[int(new_lane)]
            speed_reward = -abs(new_speed - target_speed) / self.speed_limit
            reward += speed_reward

            # Safety reward (maintain safe distance)
            same_lane = self.vehicle_states[:, 2] == new_lane
            positions = self.vehicle_states[:, 0]

            front_mask = (positions > new_position) & same_lane
            if front_mask.any():
                front_distances = positions[front_mask] - new_position
                min_front_distance = torch.min(front_distances)
                if min_front_distance < self.min_safe_distance:
                    reward += self.collision_penalty
                else:
                    reward += min_front_distance / 20.0  # Reward for safe distance

            # Progress reward
            progress_reward = (new_position - state[0]) / 10.0
            reward += progress_reward

            rewards.append(reward)

        # Check for collisions
        for i in range(self.n_vehicles):
            for j in range(i + 1, self.n_vehicles):
                if (
                    self.vehicle_states[i, 2] == self.vehicle_states[j, 2]
                    and abs(self.vehicle_states[i, 0] - self.vehicle_states[j, 0])
                    < self.min_safe_distance
                ):
                    rewards[i] += self.collision_penalty
                    rewards[j] += self.collision_penalty
                    done = True

        # Check if any vehicle reached the end
        if (self.vehicle_states[:, 0] >= self.road_length).any():
            done = True

        next_state = self.get_state()

        return next_state, torch.tensor(rewards), done, {}

    def render(self):
        """Render traffic state."""
        print("Autonomous Vehicle Coordination:")
        for i, state in enumerate(self.vehicle_states):
            print(
                f"Vehicle {i}: Pos={state[0]:.1f}, Speed={state[1]:.1f}, Lane={int(state[2])}"
            )


class SmartGridEnvironment:
    """Smart grid management with multiple agents."""

    def __init__(self, n_agents=6, n_time_steps=24, max_load=100):
        self.n_agents = n_agents  # Grid segments/agents
        self.n_time_steps = n_time_steps
        self.max_load = max_load

        # Power demand patterns (time-varying)
        self.base_demand = torch.rand(n_time_steps, n_agents) * max_load * 0.7
        self.demand_noise = torch.randn(n_time_steps, n_agents) * max_load * 0.1

        # Renewable energy availability
        self.solar_generation = (
            torch.sin(torch.linspace(0, 2 * torch.pi, n_time_steps)) * max_load * 0.3
        )
        self.wind_generation = torch.rand(n_time_steps) * max_load * 0.2

        # Current time step
        self.current_step = 0

        # Power prices
        self.peak_price = 0.5
        self.off_peak_price = 0.1

    def reset(self):
        """Reset environment."""
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        """Get current state."""
        demand = (
            self.base_demand[self.current_step] + self.demand_noise[self.current_step]
        )
        renewable_available = (
            self.solar_generation[self.current_step]
            + self.wind_generation[self.current_step]
        )

        # Time of day encoding
        time_of_day = torch.tensor(
            [
                torch.sin(torch.tensor(2 * torch.pi * self.current_step / self.n_time_steps)),
                torch.cos(torch.tensor(2 * torch.pi * self.current_step / self.n_time_steps)),
            ]
        )

        # State for each agent: [local_demand, renewable_available, time_sin, time_cos, grid_load]
        states = []
        for i in range(self.n_agents):
            state = torch.tensor(
                [
                    demand[i] / self.max_load,
                    renewable_available / self.max_load,
                    time_of_day[0],
                    time_of_day[1],
                    0.5,  # Placeholder for grid load (would be computed)
                ]
            )
            states.append(state)

        return torch.stack(states)

    def step(self, actions):
        """
        Execute power allocation actions.

        Actions: [power_generation, power_consumption_adjustment] for each agent
        """
        demand = (
            self.base_demand[self.current_step] + self.demand_noise[self.current_step]
        )
        renewable_available = (
            self.solar_generation[self.current_step]
            + self.wind_generation[self.current_step]
        )

        total_generation = 0
        total_demand = demand.sum()
        rewards = []

        for i in range(self.n_agents):
            action = actions[i]

            # Parse action
            generation = action[0] * self.max_load  # Power generation decision
            consumption_adjustment = (
                action[1] * 0.5 - 0.25
            )  # Demand response (-25% to +25%)

            # Adjusted demand
            adjusted_demand = demand[i] * (1 + consumption_adjustment)

            # Power balance for this agent
            net_power = (
                generation + renewable_available / self.n_agents - adjusted_demand
            )

            # Cost calculation
            price = (
                self.peak_price
                if self.current_step in range(8, 20)
                else self.off_peak_price
            )
            cost = abs(net_power) * price

            # Reward: negative cost + efficiency bonus
            efficiency_bonus = 0
            if net_power >= 0:  # Surplus power
                efficiency_bonus = net_power * 0.1
            else:  # Deficit
                efficiency_bonus = -abs(net_power) * 0.2

            reward = -cost + efficiency_bonus
            rewards.append(reward)

            total_generation += generation

        # Grid stability reward (shared among all agents)
        generation_balance = abs(total_generation + renewable_available - total_demand)
        stability_reward = -generation_balance * 0.05

        rewards = torch.tensor(rewards) + stability_reward

        # Update time step
        self.current_step += 1
        done = self.current_step >= self.n_time_steps

        next_state = self.get_state()

        return (
            next_state,
            rewards,
            done,
            {"total_generation": total_generation, "total_demand": total_demand},
        )

    def render(self):
        """Render grid state."""
        demand = (
            self.base_demand[self.current_step] + self.demand_noise[self.current_step]
        )
        renewable = (
            self.solar_generation[self.current_step]
            + self.wind_generation[self.current_step]
        )

        print(f"Smart Grid - Time Step {self.current_step}:")
        print(f"Total Demand: {demand.sum():.1f}")
        print(f"Renewable Available: {renewable:.1f}")
        print(f"Peak Hours: {self.current_step in range(8, 20)}")


class MultiAgentGameTheoryAnalyzer:
    """Analyzer for multi-agent game theory scenarios."""

    def __init__(self, n_agents=3, n_actions=4):
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Payoff matrices for different games
        self.payoff_matrices = self.generate_payoff_matrices()

    def generate_payoff_matrices(self):
        """Generate random payoff matrices for analysis."""
        matrices = {}

        # Prisoner's Dilemma
        pd_matrix = torch.tensor(
            [
                [[3, 3], [0, 5]],  # Both cooperate: (3,3), Cooperate-Defect: (0,5)
                [[5, 0], [1, 1]],  # Defect-Cooperate: (5,0), Both defect: (1,1)
            ]
        )
        matrices["prisoners_dilemma"] = pd_matrix

        # Battle of the Sexes
        bos_matrix = torch.tensor(
            [
                [[2, 1], [0, 0]],  # Both choose A: (2,1), A-B: (0,0)
                [[0, 0], [1, 2]],  # B-A: (0,0), Both B: (1,2)
            ]
        )
        matrices["battle_of_sexes"] = bos_matrix

        # Random game
        random_matrix = torch.rand(self.n_agents, self.n_actions, self.n_actions) * 10
        matrices["random_game"] = random_matrix

        return matrices

    def find_nash_equilibria(self, payoff_matrix, game_type="prisoners_dilemma"):
        """Find Nash equilibria in the game."""
        if game_type in ["prisoners_dilemma", "battle_of_sexes"]:
            # For 2-player games
            nash_equilibria = []

            for action1 in range(self.n_actions):
                for action2 in range(self.n_actions):
                    # Check if this is a Nash equilibrium
                    payoff1 = payoff_matrix[action1, action2, 0]
                    payoff2 = payoff_matrix[action1, action2, 1]

                    # Check if agent 1 would deviate
                    agent1_best_response = True
                    for alt_action1 in range(self.n_actions):
                        if payoff_matrix[alt_action1, action2, 0] > payoff1:
                            agent1_best_response = False
                            break

                    # Check if agent 2 would deviate
                    agent2_best_response = True
                    for alt_action2 in range(self.n_actions):
                        if payoff_matrix[action1, alt_action2, 1] > payoff2:
                            agent2_best_response = False
                            break

                    if agent1_best_response and agent2_best_response:
                        nash_equilibria.append((action1, action2))

            return nash_equilibria
        else:
            # For general games, simplified check
            return self.find_nash_general(payoff_matrix)

    def find_nash_general(self, payoff_matrix):
        """Find Nash equilibria for general games."""
        # Simplified implementation - check all action profiles
        nash_equilibria = []

        # For simplicity, assume 2-player game
        for action1 in range(self.n_actions):
            for action2 in range(self.n_actions):
                is_nash = True

                # Check if any player wants to deviate
                for player in range(2):
                    current_payoff = payoff_matrix[action1, action2, player]

                    for alt_action in range(self.n_actions):
                        if player == 0:
                            alt_payoff = payoff_matrix[alt_action, action2, player]
                        else:
                            alt_payoff = payoff_matrix[action1, alt_action, player]

                        if alt_payoff > current_payoff:
                            is_nash = False
                            break

                    if not is_nash:
                        break

                if is_nash:
                    nash_equilibria.append((action1, action2))

        return nash_equilibria

    def compute_social_welfare(self, action_profile, payoff_matrix):
        """Compute social welfare for an action profile."""
        total_payoff = 0
        for player in range(self.n_agents):
            total_payoff += payoff_matrix[
                action_profile[player], action_profile[1 - player], player
            ]
        return total_payoff

    def analyze_game(self, game_type="prisoners_dilemma"):
        """Analyze a specific game."""
        payoff_matrix = self.payoff_matrices[game_type]

        print(f"Analyzing {game_type}:")
        print(f"Payoff matrix shape: {payoff_matrix.shape}")

        # Find Nash equilibria
        nash_eq = self.find_nash_equilibria(payoff_matrix, game_type)
        print(f"Nash Equilibria: {nash_eq}")

        # Compute social welfare for each equilibrium
        for eq in nash_eq:
            welfare = self.compute_social_welfare(eq, payoff_matrix)
            print(f"Social welfare for {eq}: {welfare}")

        return {
            "nash_equilibria": nash_eq,
            "payoff_matrix": payoff_matrix,
            "game_type": game_type,
        }


# Demonstration functions
def demonstrate_resource_allocation():
    """Demonstrate resource allocation."""
    print("📊 Resource Allocation Demo")

    env = ResourceAllocationEnvironment(n_agents=3, n_resources=5)
    state = env.reset()

    # Random agent actions
    actions = torch.randint(0, 6, (3, 5))  # Random allocation requests

    next_state, rewards, done, _ = env.step(actions)

    print("Resource allocation completed:")
    print(f"Rewards: {rewards}")
    print(f"Total allocation: {next_state['allocation'].sum()}")

    env.render()

    return env


def demonstrate_autonomous_vehicles():
    """Demonstrate autonomous vehicle coordination."""
    print("\n🚗 Autonomous Vehicle Coordination Demo")

    env = AutonomousVehicleEnvironment(n_vehicles=4)
    state = env.reset()

    # Random actions: [accel, lane_change]
    actions = torch.rand(4, 2)

    next_state, rewards, done, _ = env.step(actions)

    print("Vehicle coordination step:")
    print(f"Average reward: {rewards.mean():.3f}")
    print(f"Collisions detected: {done}")

    env.render()

    return env


def demonstrate_smart_grid():
    """Demonstrate smart grid management."""
    print("\n⚡ Smart Grid Management Demo")

    env = SmartGridEnvironment(n_agents=4)
    state = env.reset()

    # Random actions: [generation, consumption_adjustment]
    actions = torch.rand(4, 2)

    next_state, rewards, done, info = env.step(actions)

    print("Smart grid step:")
    print(f"Average reward: {rewards.mean():.3f}")
    print(
        f"Generation: {info['total_generation']:.1f}, Demand: {info['total_demand']:.1f}"
    )

    env.render()

    return env


def demonstrate_game_theory():
    """Demonstrate game theory analysis."""
    print("\n🎮 Game Theory Analysis Demo")

    analyzer = MultiAgentGameTheoryAnalyzer(n_agents=2, n_actions=2)

    # Analyze Prisoner's Dilemma
    pd_analysis = analyzer.analyze_game("prisoners_dilemma")

    # Analyze Battle of the Sexes
    bos_analysis = analyzer.analyze_game("battle_of_sexes")

    print(f"Prisoner's Dilemma equilibria: {len(pd_analysis['nash_equilibria'])}")
    print(f"Battle of the Sexes equilibria: {len(bos_analysis['nash_equilibria'])}")

    return analyzer


# Run demonstrations
print("🌍 Comprehensive Multi-Agent Applications")
resource_demo = demonstrate_resource_allocation()
vehicle_demo = demonstrate_autonomous_vehicles()
grid_demo = demonstrate_smart_grid()
game_demo = demonstrate_game_theory()

print("\n🚀 Multi-agent applications ready!")
print(
    "✅ Resource allocation, autonomous vehicles, smart grid, and game theory implemented!"
)
