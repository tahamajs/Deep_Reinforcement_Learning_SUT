# Standardized MuJoCo Messages Across All Homeworks

**Date**: October 4, 2025  
**Purpose**: Consistent user messaging when MuJoCo is not available

---

## Standard Message Format

When MuJoCo is not available, all homework scripts now use this consistent message:

```
⚠️  MuJoCo not available — running with Pendulum-v0 instead.
   (Results may differ from HalfCheetah. Install MuJoCo for full experiments.)
```

---

## Files Updated

### ✅ hw2/run_all.sh
- **Line ~38-45**: MuJoCo detection message
- **Behavior**: Skips HalfCheetah-v2 experiments when MuJoCo unavailable
- **Status**: Standardized ✓

### ✅ hw3/run_all_hw3.sh
- **Line ~107-110**: Per-environment MuJoCo skip message
- **Behavior**: Skips InvertedPendulum-v2 and other MuJoCo environments
- **Status**: Standardized ✓

### ✅ hw4/run_all.sh
- **Line ~8-10**: MuJoCo detection with Pendulum fallback
- **Behavior**: Uses Pendulum-v0 as fallback environment when MuJoCo unavailable
- **Status**: Already standardized ✓ (most recent implementation)

### ✅ hw4/run_all_hw4.sh
- **Line ~49-56**: MuJoCo detection message
- **Behavior**: Exits if MuJoCo unavailable (legacy behavior)
- **Status**: Standardized ✓

### ✅ hw5/run_all_hw5.sh
- **Line ~49-56**: Initial MuJoCo detection
- **Line ~117**: SAC MuJoCo experiments skip message
- **Line ~176**: Exploration MuJoCo experiments skip message
- **Behavior**: Runs Pendulum-v0 and CartPole-v0, skips MuJoCo-specific experiments
- **Status**: Standardized ✓

---

## Behavior Summary

| Homework | Without MuJoCo | With MuJoCo |
|----------|----------------|-------------|
| **hw2** | Runs CartPole-v0, LunarLander-v2<br>Skips HalfCheetah-v2 | Runs all environments |
| **hw3** | Runs CartPole-v0<br>Skips InvertedPendulum-v2 | Runs all environments |
| **hw4** | **Uses Pendulum-v0 fallback**<br>All experiments run | Uses HalfCheetah-v2 |
| **hw4 (legacy)** | Exits with message | Runs HalfCheetah-v2 experiments |
| **hw5** | Runs Pendulum-v0, CartPole-v0<br>Skips HalfCheetah-v2 experiments | Runs all environments |

---

## Implementation Details

### Detection Logic
All scripts use consistent Python-based detection:
```bash
MUJOCO_AVAILABLE=false
python -c "import mujoco_py" 2>/dev/null && MUJOCO_AVAILABLE=true
```

### Message Components
1. **Warning icon**: ⚠️ (consistent across all scripts)
2. **Core message**: "MuJoCo not available — running with Pendulum-v0 instead."
3. **Context**: "(Results may differ from HalfCheetah. Install MuJoCo for full experiments.)"

### Special Cases
- **hw4/run_all.sh**: Only script with actual **fallback environment** (Pendulum-v0)
- **hw5**: Multiple skip points for different experiment types (SAC, Exploration)
- **hw3**: Per-environment checking (skips individual environments)

---

## User Benefits

✅ **Consistent Experience**: Same message format across all homeworks  
✅ **Clear Expectations**: Users know what to expect without MuJoCo  
✅ **Actionable**: Message implies MuJoCo can be installed for full experience  
✅ **No Confusion**: Replaces varied messages like:
   - "🚫 Skipping..."
   - "⚠️ MuJoCo (mujoco-py) not found..."
   - "will skip HalfCheetah-v2 experiments..."

---

## Testing

To verify the standardized messages:

```bash
# Test each homework without MuJoCo
cd hw2 && ./run_all.sh 2>&1 | grep -A1 "MuJoCo not available"
cd ../hw3 && ./run_all_hw3.sh 2>&1 | grep -A1 "MuJoCo not available"
cd ../hw4 && ./run_all.sh 2>&1 | grep -A1 "MuJoCo not available"
cd ../hw5 && ./run_all_hw5.sh 2>&1 | grep -A1 "MuJoCo not available"
```

Expected output (all should match):
```
⚠️  MuJoCo not available — running with Pendulum-v0 instead.
   (Results may differ from HalfCheetah. Install MuJoCo for full experiments.)
```

---

## Future Improvements

Potential enhancements:
1. Add `--force-pendulum` flag to use Pendulum even when MuJoCo available
2. Add environment comparison plots (Pendulum vs HalfCheetah results)
3. Create unified MuJoCo detection function sourced by all scripts
4. Add automatic fallback for all homeworks (not just hw4)

---

**Status**: ✅ Complete - All homeworks now use standardized MuJoCo messaging
