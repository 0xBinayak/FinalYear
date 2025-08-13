#!/usr/bin/env python3
"""
Standalone demonstration of signal processing pipeline.
This is a simple wrapper that calls the main signal processing demo.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the demo module
from src.sdr_client.demo_signal_processing import main

if __name__ == "__main__":
    main()