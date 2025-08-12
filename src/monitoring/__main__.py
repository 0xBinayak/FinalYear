"""
Main entry point for the metrics collection service.
"""

import os
import logging
import signal
import sys
from .collector import create_default_collection_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main function to run the metrics collection service."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Federated Learning Metrics Collection Service")
    
    # Get configuration from environment
    port = int(os.getenv('METRICS_PORT', '8080'))
    host = os.getenv('METRICS_HOST', '0.0.0.0')
    
    # Create and configure service
    service = create_default_collection_service()
    service.port = port
    service.host = host
    
    try:
        # Run the service
        service.run_server()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running metrics service: {e}")
        sys.exit(1)
    finally:
        logger.info("Metrics collection service stopped")

if __name__ == "__main__":
    main()