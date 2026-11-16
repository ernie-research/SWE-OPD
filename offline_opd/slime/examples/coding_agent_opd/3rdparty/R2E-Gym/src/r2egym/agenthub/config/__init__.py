"""
Config module for R2E-Gym agent hub.

This module dynamically loads scaffold names from the config directory.
"""

import os
from pathlib import Path

# Get the directory of this file
_current_dir = Path(__file__).parent

# Dynamically load all scaffold names (subdirectories in the config folder)
SCAFFOLD_NAMES = [
    item.name
    for item in _current_dir.iterdir()
    if item.is_dir() and not item.name.startswith('_') and not item.name.startswith('.')
]

# Sort for consistency
SCAFFOLD_NAMES.sort()

__all__ = ['SCAFFOLD_NAMES']
