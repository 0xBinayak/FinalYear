"""
Standalone test for federated learning functionality
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the test module
from sdr_client.test_federated_learning import main

if __name__ == "__main__":
    main()