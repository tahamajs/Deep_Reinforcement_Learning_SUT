# Quick Reference: Running All Homeworks

This guide shows you how to quickly run any homework assignment with a single command.

## TL;DR - Run Any Homework

```bash
# Navigate to any homework directory
cd homework/hw1  # or hw2, hw3, hw4, hw5

# Make script executable (first time only)
chmod +x *.sh

# Run the automation script
./run_all.sh

# That's it! Results will be in results_hw*/ directory
```

## Individual Homework Commands

### HW1: Behavioral Cloning
```bash
cd homework/hw1
./run_all.sh
# or
./run_all_environments.sh  # Alternative script
```

**Trains on**: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2, Reacher-v2, Walker2d-v2  
**Requires**: MuJoCo (most environments need it)  
**Results**: `results_hw1/`

---

### HW2: Policy Gradients
```bash
cd homework/hw2
./run_all.sh
```

**Trains on**:
- CartPole-v0 ✅ (works without MuJoCo)
- LunarLander-v2 ✅ (works without MuJoCo)
- HalfCheetah-v2 (requires MuJoCo)

**Results**: `results_hw2/logs/`, `results_hw2/plots/`, `results_hw2/videos/`

**View results**:
```bash
open results_hw2/plots/*.png
```

---

### HW3: Actor-Critic & DQN
```bash
cd homework/hw3
./run_all.sh
```

**Trains on**:
- **DQN**: LunarLander-v2 ✅ (works without MuJoCo)
- **Actor-Critic**: CartPole-v0 ✅, InvertedPendulum-v2 (requires MuJoCo)

**Results**: `results_hw3/logs/`, `results_hw3/plots/`, `results_hw3/videos/`

**View results**:
```bash
open results_hw3/plots/*.png
```

---

### HW4: Model-Based RL
```bash
cd homework/hw4
./run_all.sh
```

**Trains on**:
- HalfCheetah-v2 (requires MuJoCo)

**Note**: HW4 requires MuJoCo. Without it, the script will exit with installation instructions.

**Results**: `results_hw4/logs/`, `results_hw4/plots/`

**View results**:
```bash
open results_hw4/plots/*.png
```

---

### HW5: SAC, Exploration, Meta-Learning
```bash
cd homework/hw5
./run_all.sh
```

**Trains on**:
- **SAC**: Pendulum-v0 ✅, HalfCheetah-v2 (requires MuJoCo)
- **Exploration**: MountainCarContinuous-v0 ✅
- **Meta-Learning**: CartPole-v0 ✅

**Results**: `results_hw5/logs/`, `results_hw5/plots/`

**View results**:
```bash
open results_hw5/plots/*.png
```

---

## Run All Homeworks at Once

From the `homework/` directory:

```bash
cd homework

# Run all homeworks
./run_all_homeworks.sh

# Run specific homework(s)
./run_all_homeworks.sh --hw2-only
./run_all_homeworks.sh --hw3-only
./run_all_homeworks.sh --hw4-only
./run_all_homeworks.sh --hw5-only

# Run multiple
./run_all_homeworks.sh --hw2 --hw3 --hw5
```

---

## What Works Without MuJoCo?

| Homework | Without MuJoCo | With MuJoCo |
|----------|---------------|-------------|
| **HW1** | ⚠️ Limited | Full training |
| **HW2** | ✅ CartPole, LunarLander | + HalfCheetah |
| **HW3** | ✅ DQN + CartPole AC | + InvertedPendulum |
| **HW4** | ❌ Needs MuJoCo | Full training |
| **HW5** | ✅ Pendulum, MountainCar, CartPole | + HalfCheetah, Ant |

**Recommendation**: Start with HW2, HW3, or HW5 if you don't have MuJoCo installed yet.

---

## Results Directory Structure

After running a homework, you'll find:

```
homework/
├── hw1/
│   └── results_hw1/
│       ├── logs/              # Training logs
│       └── videos/            # Recorded videos
├── hw2/
│   └── results_hw2/
│       ├── logs/              # Training logs
│       ├── plots/             # Learning curves
│       └── videos/            # Before/after videos
├── hw3/
│   └── results_hw3/
│       ├── logs/              # Training logs
│       ├── plots/             # Learning curves
│       └── videos/            # Training videos
├── hw4/
│   └── results_hw4/
│       ├── logs/              # Training logs
│       └── plots/             # Learning curves
└── hw5/
    └── results_hw5/
        ├── logs/              # Training logs
        └── plots/             # Learning curves
```

---

## Quick Troubleshooting

### "Permission denied" when running scripts
```bash
# Make scripts executable (run once per homework)
chmod +x *.sh
./run_all.sh

# Or fix all homeworks at once from homework/ directory:
find . -name "*.sh" -type f -exec chmod +x {} \;
```

See `PERMISSION_FIX.md` for details.

### MuJoCo errors
See `MUJOCO_COMPATIBILITY.md` for detailed installation instructions, or just run the non-MuJoCo experiments:
- HW2: CartPole, LunarLander
- HW3: DQN on LunarLander, AC on CartPole
- HW5: Pendulum, MountainCar, CartPole

### TensorFlow errors
Make sure you have compatible versions:
```bash
pip install "tensorflow>=2.8.0,<2.16.0"
pip install "numpy<2.0.0"
pip install "gym>=0.21.0,<0.26.0"
```

### Viewing plots on remote server
If you're on a remote server without GUI:
```bash
# Copy plots to your local machine
scp -r user@server:~/homework/hw2/results_hw2/plots ./local_plots/

# Or use matplotlib to save as PDF
python -c "
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# ... plot code ...
plt.savefig('learning_curve.pdf')
"
```

---

## Time Estimates

Approximate running times (without MuJoCo):

| Homework | Environments | Estimated Time |
|----------|-------------|----------------|
| HW2 | 2 envs × 4 configs | ~30-60 minutes |
| HW3 | DQN + AC | ~20-40 minutes |
| HW5 | 3 algorithms | ~40-80 minutes |

**With MuJoCo** (adds HalfCheetah, etc.): expect 2-3× longer training times.

---

## Next Steps

1. **Start with HW3** - it's well-tested and has both DQN and Actor-Critic:
   ```bash
   cd homework/hw3
   ./run_all.sh
   ```

2. **Check your results**:
   ```bash
   open results_hw3/plots/*.png
   ```

3. **Try HW2** for Policy Gradients:
   ```bash
   cd homework/hw2
   ./run_all.sh
   ```

4. **Install MuJoCo** (optional) to unlock all environments:
   - See `MUJOCO_COMPATIBILITY.md`

5. **Run everything**:
   ```bash
   cd homework
   ./run_all_homeworks.sh
   ```

---

## Additional Resources

- **Complete Setup**: `COMPLETE_SETUP_GUIDE.md`
- **MuJoCo Help**: `MUJOCO_COMPATIBILITY.md`
- **Automation Details**: `AUTOMATION_GUIDE.md`
- **Individual READMEs**: Each `hw*/README.md` has algorithm-specific details

**Happy Training! 🚀**
