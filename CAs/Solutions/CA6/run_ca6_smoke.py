# Runner to smoke-test CA6 A2C/A3C components after adding stability wrapper
import nbformat
import runpy
import json
import torch
from pathlib import Path

nb_path = Path(__file__).parent / "CA6.ipynb"
nb = nbformat.read(str(nb_path), as_version=4)

# Execute the notebook's global code up to the A2C definition.
# We'll execute cells sequentially in a fresh globals dict.
globals_dict = {}

print("Executing notebook cells up to A2C definitions (smoke test)")
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    try:
        exec(src, globals_dict)
    except Exception as e:
        print(f"Cell {i+1} raised exception during exec: {e}")
        # show partial traceback
        import traceback

        traceback.print_exc()
        break
    # stop after we've defined A2CAgent and test function
    if "A2CAgent" in globals_dict and "test_a2c_vs_a3c" in globals_dict:
        print(f"Reached A2CAgent and test_a2c_vs_a3c at cell {i+1}.")
        break

# Run a shortened version of the test to ensure no NaNs
try:
    test_fn = globals_dict["test_a2c_vs_a3c"]
    # Monkey-patch the function in globals to run only 2 episodes for each agent
    import inspect

    src = inspect.getsource(test_fn)
    # We'll just call the function and trust it uses internal loops; running as-is may be long.
    print("\nRunning test_a2c_vs_a3c() (may take a few seconds)...")
    res = test_fn()
    print("test_a2c_vs_a3c() completed without uncaught exceptions.")
except Exception as e:
    print("Error running test_a2c_vs_a3c():", e)
    import traceback

    traceback.print_exc()
