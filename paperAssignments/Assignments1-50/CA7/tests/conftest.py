import os
import sys

# Ensure the CA7 package `src` is importable during tests by adding the CA7 root to sys.path
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)






