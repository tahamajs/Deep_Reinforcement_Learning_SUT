import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CommunicationChannel:
    """Communication channel for multi-agent systems."""

    def __init__(self, n_agents, message_dim=16, noise_std=0.1):
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.noise_std = noise_std
        self.message_history = []

    def send_message(self, sender_id, message, recipients=None):
        """Send message from one agent to others."""
        if recipients is None:
            recipients = list(range(self.n_agents))
            recipients.remove(sender_id)

        noisy_message = message + torch.randn_like(message) * self.noise_std

        comm_event = {
            "sender": sender_id,
            "recipients": recipients,
            "message": noisy_message,
            "timestamp": len(self.message_history),
        }

        self.message_history.append(comm_event)
        return comm_event

    def get_messages_for_agent(self, agent_id, last_n=5):
        """Get recent messages for a specific agent."""
        relevant_messages = []
        for event in self.message_history[-last_n:]:
            if agent_id in event["recipients"]:
                relevant_messages.append(
                    {
                        "sender": event["sender"],
                        "message": event["message"],
                        "timestamp": event["timestamp"],
                    }
                )
        return relevant_messages

    def clear_history(self):
        """Clear communication history."""
        self.message_history = []


class AttentionCommunication(nn.Module):
    """Attention-based communication mechanism."""

    def __init__(self, obs_dim, message_dim=16, n_heads=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.n_heads = n_heads

        self.message_encoder = nn.Sequential(
            nn.Linear(obs_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

        self.attention = nn.MultiheadAttention(message_dim, n_heads, batch_first=True)

        self.message_processor = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(self, observations, messages=None):
        """
        Args:
            observations: [batch_size, n_agents, obs_dim]
            messages: [batch_size, n_agents, message_dim] or None
        """
        batch_size, n_agents, _ = observations.shape

        encoded_messages = self.message_encoder(observations)

        if messages is not None:

            combined_messages = encoded_messages + messages
        else:
            combined_messages = encoded_messages

        attended_messages, attention_weights = self.attention(
            combined_messages, combined_messages, combined_messages
        )

        processed_messages = self.message_processor(attended_messages)

        return processed_messages, attention_weights


class CoordinationMechanism:
    """Base class for coordination mechanisms."""

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.coordination_history = []

    def coordinate(self, agent_states, task_requirements):
        """Coordinate agents based on states and task requirements."""
        raise NotImplementedError

    def evaluate_coordination(self, joint_actions, outcomes):
        """Evaluate the quality of coordination."""
        raise NotImplementedError


class MarketBasedCoordination(CoordinationMechanism):
    """Market-based coordination using auction mechanisms."""

    def __init__(self, n_agents, n_tasks=5):
        super().__init__(n_agents)
        self.n_tasks = n_tasks
        self.task_values = torch.rand(n_tasks) * 10

    def conduct_auction(self, agent_bids):
        """
        Conduct first-price sealed-bid auction.

        Args:
            agent_bids: [n_agents, n_tasks] - bid matrix

        Returns:
            task_assignments: [n_tasks] - winning agent for each task
            winning_bids: [n_tasks] - winning bid amounts
        """
        winning_agents = torch.argmax(agent_bids, dim=0)
        winning_bids = torch.max(agent_bids, dim=0).values

        return winning_agents, winning_bids

    def coordinate(self, agent_capabilities, task_requirements):
        """Coordinate using market mechanism."""

        agent_bids = torch.zeros(self.n_agents, self.n_tasks)

        for i in range(self.n_agents):
            for j in range(self.n_tasks):

                capability_match = torch.dot(
                    agent_capabilities[i], task_requirements[j]
                )
                cost = torch.norm(agent_capabilities[i] - task_requirements[j])
                agent_bids[i, j] = capability_match * self.task_values[j] - cost

        assignments, winning_bids = self.conduct_auction(agent_bids)

        coordination_result = {
            "assignments": assignments,
            "bids": agent_bids,
            "winning_bids": winning_bids,
            "total_value": torch.sum(winning_bids),
        }

        self.coordination_history.append(coordination_result)
        return coordination_result


class HierarchicalCoordination(CoordinationMechanism):
    """Hierarchical coordination with multiple levels."""

    def __init__(self, n_agents, hierarchy_levels=2):
        super().__init__(n_agents)
        self.hierarchy_levels = hierarchy_levels
        self.create_hierarchy()

    def create_hierarchy(self):
        """Create hierarchical structure."""
        self.hierarchy = {}
        agents_per_level = [self.n_agents]

        for level in range(self.hierarchy_levels):
            agents_at_level = max(1, agents_per_level[-1] // 2)
            agents_per_level.append(agents_at_level)

            self.hierarchy[level] = {
                "coordinators": list(range(agents_at_level)),
                "subordinates": list(range(agents_per_level[level])),
            }

    def coordinate_level(self, level, agent_states):
        """Coordinate agents at specific hierarchy level."""
        if level >= self.hierarchy_levels:
            return agent_states

        coordinators = self.hierarchy[level]["coordinators"]
        subordinates = self.hierarchy[level]["subordinates"]

        coordination_decisions = []
        for coordinator_id in coordinators:

            subordinate_indices = subordinates[coordinator_id :: len(coordinators)]
            if subordinate_indices:
                avg_state = torch.mean(agent_states[subordinate_indices], dim=0)
                coordination_decisions.append(avg_state)
            else:
                coordination_decisions.append(torch.zeros_like(agent_states[0]))

        return torch.stack(coordination_decisions)

    def coordinate(self, agent_states, global_objective):
        """Hierarchical coordination process."""
        current_states = agent_states
        coordination_trace = []

        for level in range(self.hierarchy_levels):
            level_decisions = self.coordinate_level(level, current_states)
            coordination_trace.append(level_decisions)
            current_states = level_decisions

        global_decision = torch.mean(current_states, dim=0)

        return {
            "global_decision": global_decision,
            "level_decisions": coordination_trace,
            "hierarchy": self.hierarchy,
        }


class EmergentCommunicationAgent(nn.Module):
    """Agent that learns to communicate through RL."""

    def __init__(self, obs_dim, action_dim, message_dim=8, vocab_size=16):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.vocab_size = vocab_size

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )

        self.message_generator = nn.Sequential(
            nn.Linear(32, message_dim), nn.ReLU(), nn.Linear(message_dim, vocab_size)
        )

        self.message_interpreter = nn.Sequential(
            nn.Linear(vocab_size, message_dim), nn.ReLU(), nn.Linear(message_dim, 16)
        )

        self.action_policy = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.value_function = nn.Sequential(
            nn.Linear(32 + 16, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def generate_message(self, obs):
        """Generate message based on observation."""
        obs_encoding = self.obs_encoder(obs)
        message_logits = self.message_generator(obs_encoding)

        message_dist = Categorical(logits=message_logits)
        message = message_dist.sample()
        message_log_prob = message_dist.log_prob(message)

        return message, message_log_prob

    def interpret_messages(self, messages):
        """Interpret received messages."""

        one_hot_messages = F.one_hot(messages, self.vocab_size).float()

        if len(one_hot_messages.shape) > 1:
            avg_message = torch.mean(one_hot_messages, dim=0)
        else:
            avg_message = one_hot_messages

        return self.message_interpreter(avg_message)

    def forward(self, obs, received_messages=None):
        """Forward pass considering observations and messages."""
        obs_encoding = self.obs_encoder(obs)

        if received_messages is not None:
            message_info = self.interpret_messages(received_messages)
            combined_input = torch.cat([obs_encoding, message_info], dim=-1)
        else:
            message_info = torch.zeros(16)
            combined_input = torch.cat([obs_encoding, message_info], dim=-1)

        action_logits = self.action_policy(combined_input)
        action_probs = F.softmax(action_logits, dim=-1)

        value = self.value_function(combined_input)

        return action_probs, value


def demonstrate_communication():
    """Demonstrate communication mechanisms."""
    print("📡 Communication Mechanisms Demo")

    comm_channel = CommunicationChannel(n_agents=4, message_dim=8)

    message = torch.randn(8)
    comm_event = comm_channel.send_message(
        sender_id=0, message=message, recipients=[1, 2, 3]
    )

    print(f"Message sent from agent 0 to agents {comm_event['recipients']}")
    print(f"Message shape: {comm_event['message'].shape}")

    messages = comm_channel.get_messages_for_agent(agent_id=1)
    print(f"Agent 1 received {len(messages)} messages")

    return comm_channel


def demonstrate_coordination():
    """Demonstrate coordination mechanisms."""
    print("\n🤝 Coordination Mechanisms Demo")

    market_coord = MarketBasedCoordination(n_agents=4, n_tasks=3)

    agent_capabilities = torch.randn(4, 5)
    task_requirements = torch.randn(3, 5)

    coordination_result = market_coord.coordinate(agent_capabilities, task_requirements)

    print("Market-based coordination result:")
    print(f"Task assignments: {coordination_result['assignments']}")
    print(f"Total value: {coordination_result['total_value']:.2f}")

    hierarchical_coord = HierarchicalCoordination(n_agents=8, hierarchy_levels=2)
    agent_states = torch.randn(8, 6)

    hierarchy_result = hierarchical_coord.coordinate(
        agent_states, global_objective=None
    )
    print(
        f"\nHierarchical coordination levels: {len(hierarchy_result['level_decisions'])}"
    )
    print(f"Global decision shape: {hierarchy_result['global_decision'].shape}")

    return market_coord, hierarchical_coord


def demonstrate_emergent_communication():
    """Demonstrate emergent communication."""
    print("\n🗣️  Emergent Communication Demo")

    agent = EmergentCommunicationAgent(
        obs_dim=10, action_dim=4, message_dim=8, vocab_size=16
    )

    obs = torch.randn(10)

    message, message_log_prob = agent.generate_message(obs)
    print(
        f"Generated message: {message.item()}, log prob: {message_log_prob.item():.3f}"
    )

    action_probs, value = agent(obs, received_messages=torch.tensor([message]))
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Value estimate: {value.item():.3f}")

    return agent


print("🌐 Communication and Coordination Systems")
comm_demo = demonstrate_communication()
coord_demo = demonstrate_coordination()
emergent_demo = demonstrate_emergent_communication()

print("\n🚀 Communication and coordination implementations ready!")
print("✅ Multi-agent communication, coordination, and emergent protocols implemented!")
