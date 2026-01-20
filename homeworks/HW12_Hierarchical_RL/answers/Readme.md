
# CQL-SAC Implementation

## Overview
This repository implements **Conservative Q-Learning Soft Actor-Critic (CQL-SAC)**, an advanced reinforcement learning algorithm that combines the benefits of Soft Actor-Critic (SAC) with Conservative Q-Learning (CQL) to address overestimation bias in Q-learning. The implementation is designed for continuous control tasks and includes comprehensive training, evaluation, and visualization capabilities.

---

## Algorithm Explanation

### Core Concepts
**Soft Actor-Critic (SAC)** is an off-policy, model-free RL algorithm that maximizes both expected reward and policy entropy. Key innovations:
- **Entropy Regularization**: Encourages exploration by maximizing policy entropy
- **Twin Q-Networks**: Two Q-functions reduce overestimation bias
- **Stochastic Policy**: Gaussian policy with tanh squashing for bounded action spaces
- **Target Networks**: Polyak-averaged target networks for stable training

**Conservative Q-Learning (CQL)** extends SAC by adding a conservative penalty to prevent Q-value overestimation:
- Ensures Q-values for policy actions are higher than for random actions
- Prevents catastrophic overestimation in offline/real-world settings
- Uses log-sum-exp to combine policy and random action Q-values

---

### Mathematical Formulation

#### SAC Objective
The actor is optimized to maximize:
$$\mathcal{L}_\pi = \mathbb{E}_{s \sim \mathcal{D}} \left[ \alpha \log \pi(a|s) - Q(s,a) \right]$$

#### CQL Penalty
The conservative penalty term:
$$\text{CQL} = \alpha \left( \mathbb{E}_{a \sim \pi} \left[ \log \sum_{a' \in \mathcal{A}} e^{Q(s,a')/\tau} \right] - \mathbb{E}_{a \sim \pi} Q(s,a) \right)$$

Where:
- $\mathcal{A}$ = Set of actions (policy + random actions)
- $\tau$ = Temperature parameter (set to 1.0)
- $\alpha$ = CQL regularization coefficient

#### Final Critic Loss
$$\mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{SAC}} + \lambda \cdot \text{CQL}$$
Where $\lambda$ = `cql_alpha` (hyperparameter)

---

## Code Structure

### Key Components
| Component | Purpose |
|-----------|---------|
| `ReplayBuffer` | Stores and samples experiences (state, action, reward, next state, done) |
| `GaussianPolicy` | Implements stochastic actor with Gaussian policy and tanh squashing |
| `QNetwork` | Twin Q-networks for Q-value estimation |
| `Config` | Holds all hyperparameters for training |

### Training Workflow
1. **Initialization**: Set seeds, initialize networks, optimizers
2. **Random Exploration**: Collect initial experiences with random actions
3. **Experience Collection**: Store transitions in replay buffer
4. **Training Loop**:
   - Sample batch from replay buffer
   - Compute SAC Q-learning loss
   - Add CQL penalty to critic loss
   - Update actor to maximize entropy-regularized Q-values
   - Update target networks with polyak averaging
5. **Evaluation**: Periodically evaluate policy performance

---

## Installation

```bash
# Install required dependencies
pip install gym torch numpy matplotlib
```

---

## Usage

### Training
```bash
python cql_sac.py --epochs 100 --steps-per-epoch 1000 --seed 0
```

### Command-Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--steps-per-epoch` | 1000 | Steps per epoch |
| `--seed` | 0 | Random seed for reproducibility |

---

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env_name` | "Pendulum-v1" | Gym environment name |
| `seed` | 0 | Random seed |
| `epochs` | 100 | Training epochs |
| `steps_per_epoch` | 1000 | Steps per epoch |
| `start_steps` | 1000 | Random exploration steps |
| `update_after` | 1000 | Steps before starting updates |
| `update_every` | 50 | Steps between updates |
| `batch_size` | 256 | Batch size for training |
| `gamma` | 0.99 | Discount factor |
| `polyak` | 0.995 | Target network update rate |
| `lr` | 3e-4 | Learning rate |
| `alpha` | 0.2 | Entropy coefficient |
| `cql_alpha` | 1.0 | CQL regularization coefficient |
| `device` | "cpu" | Training device (cpu/cuda) |
| `save_fig` | "results/cql_sac_training.png" | Path to save training plot |

---

## Results

After training, the script generates:
1. **Training Plot**: Moving average reward (100 episodes) saved to `results/cql_sac_training.png`
2. **Evaluation Metrics**: Mean return over 5 deterministic evaluation episodes

### Example Output
```
Epoch    Step     Avg Reward    Critic Loss    Actor Loss
-----------------------------------------------------
1       1000    -1107.35       22.48          5.59           
1       1500    -1150.24       8.75           17.99          
2       2000    -1312.59       5.40           31.32          
2       2500    -1360.48       7.33           49.89          
3       3000    -1407.68       9.46           66.12          
3       3500    -1411.27       73.87          79.98          
4       4000    -1388.20       7.32           94.41          
4       4500    -1367.05       130.63         104.26         
5       5000    -1312.23       104.97         109.18         
5       5500    -1239.21       170.52         120.16         
6       6000    -1179.92       11.24          125.68         
6       6500    -1117.97       504.92         123.95         
7       7000    -1044.06       438.17         132.90         
7       7500    -997.82        382.89         131.44         
8       8000    -950.43        11.79          125.37         
8       8500    -920.68        15.72          132.18         
9       9000    -876.43        133.49         144.69         
9       9500    -847.50        16.76          130.70         
10      10000   -799.13        13.98          121.49         
10      10500   -778.81        10.06          142.76         
11      11000   -756.51        191.59         129.81         
11      11500   -738.89        505.50         127.39         
12      12000   -719.22        329.65         132.28         
12      12500   -704.21        12.40          142.52         
13      13000   -681.54        17.74          126.98         
13      13500   -661.22        15.10          120.12         
14      14000   -640.20        704.72         128.72         
14      14500   -624.17        14.13          121.74         
15      15000   -606.34        11.89          126.64         
15      15500   -593.86        16.45          120.42         
16      16000   -574.93        128.04         128.46         
16      16500   -565.49        798.46         124.31         
17      17000   -552.90        23.06          118.63         
17      17500   -540.20        29.51          101.97         
18      18000   -526.26        17.88          110.93         
18      18500   -518.87        24.55          110.07         
19      19000   -510.13        13.25          129.16         
19      19500   -504.72        575.98         125.79         
20      20000   -497.11        38.43          114.90         
20      20500   -474.55        1431.39        109.53         
21      21000   -450.42        1443.22        98.17          
21      21500   -428.97        28.81          105.85         
22      22000   -381.90        21.47          99.80          
22      22500   -352.33        43.71          92.42          
23      23000   -308.16        14.08          109.21         
23      23500   -286.16        244.61         90.68          
24      24000   -252.19        14.98          92.10          
24      24500   -231.60        8.59           88.08          
25      25000   -209.64        15.85          92.11          
25      25500   -208.55        17.75          94.22          
26      26000   -192.91        10.01          90.97          
26      26500   -190.38        28.73          102.45         
27      27000   -187.77        19.57          102.11         
27      27500   -189.27        38.60          95.80          
28      28000   -184.52        26.55          84.87          
28      28500   -183.07        11.30          73.59          
29      29000   -179.01        24.11          86.14          
29      29500   -180.15        20.46          92.97          
30      30000   -181.43        14.25          91.17          
30      30500   -178.53        19.41          90.33          
31      31000   -173.44        11.07          83.73          
31      31500   -169.57        19.16          91.29          
32      32000   -162.99        24.62          97.88          
32      32500   -160.39        29.50          97.64          
33      33000   -158.91        19.88          85.29          
33      33500   -161.52        36.72          80.54          
34      34000   -162.62        37.06          86.21          
34      34500   -165.11        10.95          74.27          
35      35000   -162.21        324.16         83.10          
35      35500   -163.67        730.97         80.95          
36      36000   -166.14        26.50          88.83          
36      36500   -164.85        28.04          85.50          
37      37000   -162.19        54.61          73.05          
37      37500   -166.02        30.29          83.31          
38      38000   -167.19        58.43          78.40          
38      38500   -166.00        35.94          80.58          
39      39000   -164.92        152.46         74.38          
39      39500   -163.57        1378.84        78.65          
40      40000   -160.90        6.80           78.38          
40      40500   -158.55        22.27          78.77          
41      41000   -160.24        32.31          72.01          
41      41500   -160.11        1160.49        82.61          
42      42000   -160.26        13.54          80.33          
42      42500   -160.18        34.78          74.80          
43      43000   -160.18        31.95          73.81          
43      43500   -157.38        10.15          77.79          
44      44000   -157.35        12.31          87.11          
44      44500   -158.87        21.15          75.91          
45      45000   -158.67        1270.33        80.45          
45      45500   -156.92        60.20          76.78          
46      46000   -159.67        11.06          75.20          
46      46500   -162.25        20.83          86.89          
47      47000   -162.01        6.56           82.05          
47      47500   -159.24        11.27          77.64          
48      48000   -155.61        5.60           72.02          
48      48500   -153.00        1293.68        75.66          
49      49000   -151.86        1297.58        68.41          
49      49500   -149.34        5.23           81.50          
50      50000   -152.79        20.29          76.36          
50      50500   -155.11        5.75           69.13          
51      51000   -151.63        21.42          71.31          
51      51500   -151.71        8.32           74.37          
52      52000   -153.02        38.83          78.80          
52      52500   -153.10        6.62           75.62          
53      53000   -150.82        24.35          72.33          
53      53500   -151.82        23.82          79.67          
54      54000   -148.17        6.32           76.43          
54      54500   -146.87        106.85         73.95          
55      55000   -149.50        33.89          75.06          
55      55500   -152.37        31.07          72.22          
56      56000   -152.10        23.43          73.67          
56      56500   -152.06        25.90          82.65          
57      57000   -154.46        19.83          82.27          
57      57500   -153.09        292.88         75.91          
58      58000   -155.30        5.82           75.95          
58      58500   -154.06        20.15          71.21          
59      59000   -154.09        34.30          75.63          
59      59500   -154.13        5.82           67.35          
60      60000   -153.09        51.05          75.54          
60      60500   -154.40        68.31          75.74          
61      61000   -155.21        6.00           71.86          
61      61500   -155.39        38.45          74.54          
62      62000   -158.37        50.35          75.34          
62      62500   -160.59        435.65         69.80          
63      63000   -161.92        23.48          78.78          
63      63500   -161.58        52.33          72.43          
64      64000   -162.89        58.20          80.41          
64      64500   -160.13        20.91          64.42          
65      65000   -157.54        34.85          73.06          
65      65500   -156.34        3.07           66.04          
66      66000   -152.34        5.78           76.09          
66      66500   -149.82        4.39           67.44          
67      67000   -149.99        345.20         73.29          
67      67500   -149.87        6.63           74.33          
68      68000   -149.91        5.91           75.85          
68      68500   -151.11        21.58          73.24          
69      69000   -153.57        3.08           68.86          
69      69500   -153.51        21.12          71.92          
70      70000   -152.54        4.57           69.72          
70      70500   -151.54        33.84          72.79          
71      71000   -150.32        50.68          70.73          
71      71500   -149.05        38.57          68.14          
72      72000   -147.66        5.04           76.46          
72      72500   -148.83        22.45          74.85          
73      73000   -152.34        10.40          70.31          
73      73500   -151.20        4.22           69.77          
74      74000   -153.52        53.94          71.98          
74      74500   -153.61        35.33          72.89          
75      75000   -151.08        8.42           76.65          
75      75500   -149.01        40.50          70.10          
76      76000   -147.80        35.55          71.81          
76      76500   -149.05        6.50           68.28          
77      77000   -147.84        6.19           78.31          
77      77500   -147.89        5.95           70.64          
78      78000   -147.05        6.21           74.26          
78      78500   -149.25        5.74           74.84          
79      79000   -146.76        22.16          67.64          
79      79500   -149.27        38.23          73.85          
80      80000   -150.56        7.88           75.18          
80      80500   -151.53        40.49          75.67          
81      81000   -147.84        22.76          69.45          
81      81500   -146.64        10.37          70.02          
82      82000   -143.72        490.76         73.48          
82      82500   -141.62        4.55           69.50          
83      83000   -139.15        41.88          68.60          
83      83500   -138.10        6.22           71.68          
84      84000   -135.58        23.37          62.61          
84      84500   -136.81        20.80          65.34          
85      85000   -136.83        19.65          72.20          
85      85500   -138.81        5.82           74.33          
Mean agent return (deterministic): -10.25
```

---

## Key Implementation Details

### CQL Penalty Calculation
```python
# Sample random actions
sample_actions = torch.rand((batch_size, 10, act_dim), device=device) * 2 * act_limit - act_limit

# Compute Q-values for random actions
q1_rand = critic1(obs_repeat, sample_actions).view(batch_size, -1)
q2_rand = critic2(obs_repeat, sample_actions).view(batch_size, -1)

# Combine with policy actions
cat_q1 = torch.cat([q1_rand, q1.unsqueeze(1)], dim=1)
cat_q2 = torch.cat([q2_rand, q2.unsqueeze(1)], dim=1)

# Compute CQL penalty
cql_q1 = torch.logsumexp(cat_q1 / 1.0, dim=1).mean() - q1.mean()
cql_q2 = torch.logsumexp(cat_q2 / 1.0, dim=1).mean() - q2.mean()
cql_penalty = cql_alpha * (cql_q1 + cql_q2)
```

### Policy Action Sampling
```python
def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu, log_std = self.forward(obs)
    std = log_std.exp()
    pi_dist = Normal(mu, std)
    x_t = pi_dist.rsample()
    y_t = torch.tanh(x_t)
    action = y_t * self.act_limit
    log_prob = pi_dist.log_prob(x_t).sum(axis=-1) - (2 * (math.log(2) - x_t - F.softplus(-2 * x_t))).sum(axis=-1)
    return action, log_prob
```

---

## References

1. Haarnoja, T., et al. (2018). [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
2. Kumar, A., et al. (2020). [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
3. ZharfaTech (2026). *CQL-SAC Implementation for Continuous Control Tasks*

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.