import os
import sys

# Ensure the repository root (parent of this `testing/` dir) is on sys.path
# so tests can import the `src` package when pytest is invoked from anywhere.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
