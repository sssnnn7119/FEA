#!/usr/bin/env python
"""
Launcher script for the FEA UI application
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui import launch_ui

if __name__ == "__main__":
    launch_ui()