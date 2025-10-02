from .base_agent import BaseAgent
from .sac_agent import SACAgent
from .mb_agent import MBAgent
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *

class MBPOAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBPOAgent, self).__init__()
        self.mb_agent = MBAgent(env, agent_params)
        self.sac_agent = SACAgent(env, agent_params['sac_params'])
        self.env = env

        self.actor = self.sac_agent.actor

    def train(self, *args):
        return self.mb_agent.train(*args)

    def train_sac(self, *args):
        return self.sac_agent.train(*args)

    def collect_model_trajectory(self, rollout_length=1):
        ob, _, _, _, terminal = TODO

        obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
        for _ in range(rollout_length):

            ac = TODO
            next_ob = TODO
            rew, _ = TODO

            obs.append(ob[0])
            acs.append(ac[0])
            rewards.append(rew[0])
            next_obs.append(next_ob[0])
            terminals.append(terminal[0])

            ob = next_ob
        return [Path(obs, image_obs, acs, rewards, next_obs, terminals)]

    def add_to_replay_buffer(self, paths, from_model=False, **kwargs):
        self.sac_agent.add_to_replay_buffer(paths)

        if not from_model:
            self.mb_agent.add_to_replay_buffer(paths, **kwargs)

    def sample(self, *args, **kwargs):
        return self.mb_agent.sample(*args, **kwargs)

    def sample_sac(self, *args, **kwargs):
        return self.sac_agent.sample(*args, **kwargs)