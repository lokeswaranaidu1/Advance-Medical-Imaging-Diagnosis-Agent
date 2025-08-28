#!/usr/bin/env python3
"""
Startup script for Advanced Medical Imaging Diagnosis Agent FastAPI
"""

import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    try:
        logger.info("Starting Advanced Medical Imaging Diagnosis Agent FastAPI...")
        
        # Check if required environment variables are set
        required_env_vars = ['OPENAI_API_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            logger.warning("Some features may not work properly")
        
        # Import and validate configuration
        try:
            from config import config
            config_issues = config.validate_config()
            if config_issues:
                logger.warning(f"Configuration issues found: {config_issues}")
        except Exception as e:
            logger.error(f"Configuration error: {str(e)}")
            logger.warning("Using default configuration")
        
        # Start FastAPI server
        uvicorn.run(
            "api:app",
            host="0.0.0.0",  # Bind to all interfaces
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
