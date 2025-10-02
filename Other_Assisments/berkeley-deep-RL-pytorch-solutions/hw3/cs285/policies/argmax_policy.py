import torch
class ArgMaxPolicy:

    def __init__(self, critic, device):
        self.critic = critic
        self.device = device

    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            observation = torch.tensor(obs[None], dtype=torch.float32).to(self.device)

        return self.critic.Q_func(observation).squeeze().argmax().item()