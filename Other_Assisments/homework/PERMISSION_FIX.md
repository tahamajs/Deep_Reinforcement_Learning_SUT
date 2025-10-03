# Quick Fix: Permission Denied Errors

## Problem
When trying to run `./run_all.sh` in any homework directory, you might see:

```
Permission denied
cannot execute: Undefined error: 0
```

This happens because the shell scripts don't have execute permissions.

## Solution

Make the scripts executable using `chmod +x`:

### Fix All Homeworks at Once

```bash
cd /Users/tahamajs/Documents/uni/DRL/Other_Assisments/homework

# Make all .sh files executable
chmod +x hw2/*.sh
chmod +x hw3/*.sh
chmod +x hw4/*.sh
chmod +x hw5/*.sh
chmod +x *.sh
```

### Fix Individual Homework

```bash
# HW2
cd homework/hw2
chmod +x *.sh
./run_all.sh

# HW3
cd homework/hw3
chmod +x *.sh
./run_all.sh

# HW4
cd homework/hw4
chmod +x *.sh
./run_all.sh

# HW5
cd homework/hw5
chmod +x *.sh
./run_all.sh
```

## What `chmod +x` Does

- `chmod` = "change mode" (permissions)
- `+x` = add execute permission
- `*.sh` = all files ending in .sh

Without execute permission, the shell can't run the script, even if it's a valid bash script.

## Quick One-Liner

Run this from the `homework` directory to fix all scripts:

```bash
find . -name "*.sh" -type f -exec chmod +x {} \;
```

This finds all `.sh` files and makes them executable.

## Verify Permissions

Check if a script is executable:

```bash
ls -la run_all.sh

# Should show something like:
# -rwxr-xr-x  1 user  staff  123  Oct  4 20:00 run_all.sh
#  ^^^
#  These x's mean executable
```

## After Making Executable

Just run the script normally:

```bash
./run_all.sh
```

No more "Permission denied" errors! ✅
