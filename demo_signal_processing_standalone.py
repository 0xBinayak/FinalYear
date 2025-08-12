"""
Standalone demonstration of signal processing pipeline
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the demo module
from sdr_client.demo_signal_processing import main

if __name__ == "__main__":
    main()