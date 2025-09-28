import nbformat
import runpy
import json
import torch
from pathlib import Path

nb_path = Path(__file__).parent / "CA6.ipynb"
nb = nbformat.read(str(nb_path), as_version=4)


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

        import traceback

        traceback.print_exc()
        break

    if "A2CAgent" in globals_dict and "test_a2c_vs_a3c" in globals_dict:
        print(f"Reached A2CAgent and test_a2c_vs_a3c at cell {i+1}.")
        break


try:
    test_fn = globals_dict["test_a2c_vs_a3c"]

    import inspect

    src = inspect.getsource(test_fn)

    print("\nRunning test_a2c_vs_a3c() (may take a few seconds)...")
    res = test_fn()
    print("test_a2c_vs_a3c() completed without uncaught exceptions.")
except Exception as e:
    print("Error running test_a2c_vs_a3c():", e)
    import traceback

    traceback.print_exc()
