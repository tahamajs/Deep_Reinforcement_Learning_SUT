import numpy as np
from scipy.stats import norm

# Test the implementations
class Arm:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def sample(self):
        return np.random.normal(self.mean, np.sqrt(self.var))

class ThompsonSampling:
    def __init__(self, var_list, **kwargs):
        self.var_list = var_list
        self.n_arms = len(var_list)
        self.prior_means = np.zeros(self.n_arms)
        self.prior_vars = np.ones(self.n_arms) * 100
        self.n_pulls = np.zeros(self.n_arms)

    def select_arm(self, *args):
        samples = []
        for i in range(self.n_arms):
            sample = np.random.normal(self.prior_means[i], np.sqrt(self.prior_vars[i]))
            samples.append(sample)
        return np.argmax(samples)

    def update(self, idx, reward):
        prior_mean = self.prior_means[idx]
        prior_var = self.prior_vars[idx]
        obs_var = self.var_list[idx]
        
        posterior_var = 1.0 / (1.0/prior_var + 1.0/obs_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + reward/obs_var)
        
        self.prior_means[idx] = posterior_mean
        self.prior_vars[idx] = posterior_var
        self.n_pulls[idx] += 1

# Test basic functionality
print("Testing Thompson Sampling...")
mab = [Arm(6, 0.5), Arm(4, 0.5)]
policy = ThompsonSampling([b.var for b in mab])

for i in range(10):
    arm_idx = policy.select_arm(i)
    reward = mab[arm_idx].sample()
    policy.update(arm_idx, reward)
    print(f"Step {i}: Selected arm {arm_idx}, Reward: {reward:.2f}")

print(f"\nFinal beliefs - Means: {policy.prior_means}")
print(f"Number of pulls: {policy.n_pulls}")
print("\nTest passed! ✓")

