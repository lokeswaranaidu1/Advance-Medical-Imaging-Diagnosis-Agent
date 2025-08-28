#!/usr/bin/env python3
"""
Startup script for Advanced Medical Imaging Diagnosis Agent Streamlit Application
"""

import streamlit as st
import logging
import sys
import os
from pathlib import Path
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streamlit.log')
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("Checking dependencies...")
    
    required_modules = [
        'streamlit',
        'fastapi',
        'torch',
        'opencv-python',
        'pydicom',
        'nibabel',
        'openai',
        'sqlalchemy',
        'reportlab'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        logger.error("Please install missing dependencies:")
        logger.error("pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are available")
    return True

def check_configuration():
    """Check if configuration is properly set up"""
    logger.info("Checking configuration...")
    
    # Check if config file exists
    config_file = project_root / "config.py"
    if not config_file.exists():
        logger.error("Configuration file not found: config.py")
        return False
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        logger.warning(".env file not found, using default configuration")
        logger.info("Copy env_template.txt to .env and configure your settings")
    
    # Check required environment variables
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Some features may not work properly")
    
    logger.info("Configuration check completed")
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    logger.info("Creating necessary directories...")
    
    directories = [
        "uploads",
        "models",
        "reports",
        "cache",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Directory ready: {directory}")

def start_streamlit():
    """Start the Streamlit application"""
    logger.info("Starting Streamlit application...")
    
    try:
        # Check if app.py exists
        app_file = project_root / "app.py"
        if not app_file.exists():
            logger.error("Main application file not found: app.py")
            return False
        
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Start Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        logger.info(f"Starting Streamlit with command: {' '.join(cmd)}")
        
        # Run Streamlit
        process = subprocess.run(cmd, cwd=project_root)
        
        if process.returncode == 0:
            logger.info("Streamlit application started successfully")
            return True
        else:
            logger.error(f"Streamlit application failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error starting Streamlit: {str(e)}")
        return False

def main():
    """Main startup function"""
    try:
        logger.info("🏥 Advanced Medical Imaging Diagnosis Agent - Streamlit Startup")
        logger.info("=" * 60)
        
        # Step 1: Check dependencies
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            sys.exit(1)
        
        # Step 2: Check configuration
        if not check_configuration():
            logger.error("❌ Configuration check failed")
            sys.exit(1)
        
        # Step 3: Create directories
        create_directories()
        
        # Step 4: Start Streamlit
        logger.info("🚀 Starting Streamlit application...")
        if start_streamlit():
            logger.info("✅ Streamlit application started successfully")
            logger.info("🌐 Access your application at: http://localhost:8501")
        else:
            logger.error("❌ Failed to start Streamlit application")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
