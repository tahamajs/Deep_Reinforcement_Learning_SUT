import sys
import os
# Ensure the local src/ directory is importable when running tests from the CA12 folder
HERE = os.path.dirname(__file__)
PKG = os.path.abspath(os.path.join(HERE, "..", "src"))
if PKG not in sys.path:
    sys.path.insert(0, PKG)
