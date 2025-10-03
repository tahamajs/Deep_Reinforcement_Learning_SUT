# HW4 MuJoCo Import Fix

## Problem

HW4 was crashing immediately when run without MuJoCo installed, even though we had MuJoCo detection code. The issue was the **import order**:

### Before (Broken):
```python
import tensorflow as tf
from half_cheetah_env import HalfCheetahEnv  # ❌ This imports mujoco_py IMMEDIATELY
from logger import logger
from model_based_rl import ModelBasedRL

# ... later ...
# MuJoCo check happens HERE (too late!)
if args.env in MUJOCO_ENVS:
    try:
        import mujoco_py
    except ImportError:
        print("MuJoCo not installed")
```

**Problem**: `half_cheetah_env.py` imports `mujoco_py` at the top level:
```python
# half_cheetah_env.py
from gym.envs.mujoco import mujoco_env  # ❌ Crashes here if no MuJoCo
```

So the script crashed during imports, before reaching the MuJoCo check!

### After (Fixed):
```python
import tensorflow as tf

# Define MuJoCo check function
def check_mujoco_available():
    try:
        import mujoco_py
        return True
    except (ImportError, Exception):
        return False

# Parse arguments
args = parser.parse_args()

# Check for MuJoCo BEFORE importing environment
if args.env in MUJOCO_ENVS and not check_mujoco_available():
    print("❌ MuJoCo not installed")
    sys.exit(1)

# NOW import MuJoCo-dependent modules (after the check)
from half_cheetah_env import HalfCheetahEnv  # ✅ Safe now
from logger import logger
from model_based_rl import ModelBasedRL
```

## Solution

**Key Change**: Move the MuJoCo check to happen BEFORE importing any MuJoCo-dependent modules.

1. Import only standard library and TensorFlow at the top
2. Define MuJoCo check function
3. Parse command-line arguments
4. Check if MuJoCo is available
5. If not available, show helpful message and exit cleanly
6. If available, THEN import the environment modules

## Result

Now when you run HW4 without MuJoCo:

```bash
./run_all.sh

# Output:
❌ ERROR: Environment 'HalfCheetah' requires MuJoCo, but it's not installed.

📦 To install MuJoCo:
   1. Download MuJoCo 2.1.0 from: https://github.com/deepmind/mujoco/releases
   2. Extract to ~/.mujoco/mujoco210
   3. Install mujoco-py: pip install mujoco-py
   4. On macOS: brew install gcc

Exiting HW4 (all experiments require MuJoCo).
```

Clean exit! No scary traceback. 🎉

## Technical Details

### Python Import Behavior

Python executes imports immediately when it encounters them:

```python
# When Python sees this line:
from half_cheetah_env import HalfCheetahEnv

# It does this:
# 1. Find half_cheetah_env.py
# 2. Execute ALL code at module level in that file
# 3. In half_cheetah_env.py, it sees:
#    from gym.envs.mujoco import mujoco_env
# 4. Execute gym/envs/mujoco/__init__.py
# 5. That tries: import mujoco_py
# 6. CRASH if mujoco-py not installed!
```

### The Fix: Lazy Imports

By moving imports AFTER the check, we implement "lazy imports":

```python
# Don't import yet...
if mujoco_available:
    from half_cheetah_env import HalfCheetahEnv  # Only import if safe
else:
    print("Can't import - MuJoCo missing")
    sys.exit(1)
```

## Files Changed

1. **`main.py`**:
   - Moved MuJoCo-dependent imports to after the availability check
   - Added `sys` import for clean exit
   - Improved error messages with emojis and clear instructions

## Testing

### Without MuJoCo (Your Case):
```bash
cd homework/hw4
./run_all.sh

# ✅ Clean exit with helpful message
# ✅ No scary tracebacks
# ✅ Clear installation instructions
```

### With MuJoCo:
```bash
cd homework/hw4
./run_all.sh

# ✅ Runs all experiments normally
# ✅ Trains on HalfCheetah-v2
# ✅ Generates plots and results
```

## Why This Matters

**Good Error Handling Principles**:
1. ✅ Fail fast with clear messages
2. ✅ Tell user exactly what's wrong
3. ✅ Provide actionable solution
4. ✅ No scary stack traces for expected conditions

**Bad**: 
```
RuntimeError: Could not find GCC 6 or GCC 7 executable.
  File "/path/to/mujoco_py/builder.py", line 301, in _build_impl
  File "/path/to/mujoco_py/builder.py", line 202, in build
  ...
(User has no idea what went wrong or how to fix it)
```

**Good**:
```
❌ ERROR: Environment 'HalfCheetah' requires MuJoCo, but it's not installed.

📦 To install MuJoCo:
   1. Download MuJoCo 2.1.0 from: https://github.com/deepmind/mujoco/releases
   2. Extract to ~/.mujoco/mujoco210
   3. Install mujoco-py: pip install mujoco-py
   4. On macOS: brew install gcc
```

Clear, actionable, friendly! ✨

## Related Files

- `MUJOCO_COMPATIBILITY.md` - Comprehensive MuJoCo setup guide
- `run_all.sh` - Automation script with MuJoCo detection
- `README.md` - Updated with MuJoCo requirements

## Summary

**Problem**: Import-time crash when MuJoCo not available  
**Solution**: Check for MuJoCo before importing MuJoCo-dependent modules  
**Result**: Clean, helpful error message instead of scary traceback  

Now you can run HW4's script and get a friendly message instead of a crash! 🎉
