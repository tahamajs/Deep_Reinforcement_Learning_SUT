# MuJoCo Setup Guide for macOS

## The Problem

You're getting: `RuntimeError: Could not find GCC 6 or GCC 7 executable`

This happens because `mujoco-py==1.50.1.56` (old version) requires specific GCC versions and has compatibility issues with modern macOS (especially M1/M2 chips and Python 3.13).

## Solution Options

### Option 1: Use Modern Versions (RECOMMENDED) ⭐

Upgrade to newer versions of gym and mujoco that work with modern systems:

```bash
# Deactivate current environment
deactivate

# Remove old environment
rm -rf venv

# Create new environment with Python 3.9 (more compatible)
python3.9 -m venv venv
source venv/bin/activate

# Install modern versions
pip install --upgrade pip
pip install gym[mujoco]
pip install tensorflow
pip install numpy matplotlib seaborn

# Or use the newer gymnasium
pip install gymnasium[mujoco]
```

**Note:** This requires updating the code to use newer gym/gymnasium API.

---

### Option 2: Install GCC for Old MuJoCo (Complex)

If you must use the old version:

```bash
# Install GCC 7
brew install gcc@7

# Create symlinks
sudo ln -s /usr/local/bin/gcc-7 /usr/local/bin/gcc
sudo ln -s /usr/local/bin/g++-7 /usr/local/bin/g++

# Try installation again
pip install mujoco-py==1.50.1.56
```

⚠️ **Warning:** This may still fail on M1/M2 Macs or newer macOS versions.

---

### Option 3: Use Docker (Most Reliable)

Run everything in a Linux container:

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3.6 python3-pip \
    gcc g++ \
    libglew-dev \
    libosmesa6-dev \
    patchelf

WORKDIR /workspace
COPY . .

RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
EOF

# Build and run
docker build -t drl-hw1 .
docker run -it -v $(pwd):/workspace drl-hw1
```

---

### Option 4: Use Google Colab (EASIEST) ⭐⭐⭐

Upload your code to Google Colab and run there:

1. Go to https://colab.research.google.com
2. Upload your files
3. Run this setup:

```python
# Install dependencies
!pip install gym==0.10.5
!pip install tensorflow
!apt-get install -y python-opengl
!apt-get install -y xvfb
!pip install pyvirtualdisplay

# Install MuJoCo
!mkdir -p /root/.mujoco
!wget https://www.roboti.us/download/mujoco150_linux.zip
!unzip mujoco150_linux.zip -d /root/.mujoco/
!mv /root/.mujoco/mujoco150_linux /root/.mujoco/mujoco150

# Add your license key (you need to get this from MuJoCo website)
# !echo "YOUR_LICENSE_KEY" > /root/.mujoco/mjkey.txt

!pip install mujoco-py==1.50.1.56

# Run your code
!python run_full_pipeline.py --env Hopper-v2
```

---

### Option 5: Skip MuJoCo, Use Simple Environments

Modify the code to use non-MuJoCo environments for testing:

```bash
# These don't require MuJoCo
- CartPole-v1
- LunarLander-v2 (requires box2d: pip install box2d-py)
- MountainCar-v0
- Acrobot-v1
```

---

## Quick Fix: Update to Modern Stack

Let me create an updated version that works with modern libraries:

### Step 1: Update Requirements

```bash
# Create new requirements-modern.txt
cat > requirements-modern.txt << EOF
gymnasium>=0.29.0
gymnasium[mujoco]
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
EOF
```

### Step 2: Install Modern Packages

```bash
pip install -r requirements-modern.txt
```

### Step 3: Code Changes Needed

The main changes needed:
1. `gym` → `gymnasium`
2. `env.spec.timestep_limit` → `env.spec.max_episode_steps`
3. Update environment names (mostly same)

---

## My Recommendation

**For macOS users (especially M1/M2):**

1. **Best:** Use Google Colab (easiest, no local setup)
2. **Good:** Use Docker with Linux container
3. **Alternative:** Update to modern gym/gymnasium

**For Linux users:**
- Install gcc and follow Option 2

**For Windows users:**
- Use WSL2 + Docker or Google Colab

---

## What Should You Do Now?

### Quick Test without MuJoCo

Let me create a version that works without MuJoCo for testing:

```bash
# Install simple environment
pip install box2d-py

# Test with LunarLander (no MuJoCo needed)
python run_full_pipeline.py --env LunarLander-v2
```

But you'll need to add the expert policy for LunarLander first.

---

## Alternative: I Can Help You

I can:

1. ✅ **Create a Google Colab notebook** with everything set up
2. ✅ **Update code for modern gymnasium** (recommended)
3. ✅ **Create a Docker setup** for Linux environment
4. ✅ **Provide a simpler test** with non-MuJoCo environments

Which would you prefer?

---

## Summary

**The core issue:** Old MuJoCo (1.50) doesn't work well on modern macOS.

**Best solutions:**
1. Use Google Colab (fastest to get running)
2. Update to modern gymnasium/mujoco (best long-term)
3. Use Docker (most reliable cross-platform)

**Your code is correct** - it's just the environment setup that needs adjustment for modern systems!
