#!/usr/bin/env python3
"""
Start both Streamlit and FastAPI applications for Advanced Medical Imaging Diagnosis Agent
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import logging
import threading
import requests
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApplicationManager:
    """Manages starting and stopping of both applications"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes: List[subprocess.Popen] = []
        self.running = False
        
        # Application configurations
        self.apps = {
            'streamlit': {
                'script': 'start_streamlit.py',
                'port': 8501,
                'health_url': 'http://localhost:8501',
                'name': 'Streamlit Frontend'
            },
            'api': {
                'script': 'start_api.py',
                'port': 8000,
                'health_url': 'http://localhost:8000/health',
                'name': 'FastAPI Backend'
            }
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")
        
        # Check if required files exist
        required_files = ['start_streamlit.py', 'start_api.py', 'config.py']
        for file in required_files:
            if not (self.project_root / file).exists():
                logger.error(f"Required file not found: {file}")
                return False
        
        # Check if ports are available
        for app_name, config in self.apps.items():
            if not self._is_port_available(config['port']):
                logger.warning(f"Port {config['port']} is already in use")
        
        logger.info("Prerequisites check completed")
        return True
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def start_application(self, app_name: str) -> Optional[subprocess.Popen]:
        """Start a specific application"""
        if app_name not in self.apps:
            logger.error(f"Unknown application: {app_name}")
            return None
        
        app_config = self.apps[app_name]
        script_path = self.project_root / app_config['script']
        
        logger.info(f"Starting {app_config['name']} on port {app_config['port']}...")
        
        try:
            # Start the application
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"{app_config['name']} started successfully (PID: {process.pid})")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Failed to start {app_config['name']}")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting {app_config['name']}: {str(e)}")
            return None
    
    def start_all_applications(self) -> bool:
        """Start all applications"""
        logger.info("Starting all applications...")
        
        if not self.check_prerequisites():
            return False
        
        # Start applications in sequence
        for app_name in self.apps.keys():
            process = self.start_application(app_name)
            if process:
                self.processes.append(process)
                logger.info(f"Started {app_name}")
            else:
                logger.error(f"Failed to start {app_name}")
                self.stop_all_applications()
                return False
        
        self.running = True
        logger.info("All applications started successfully!")
        
        # Start health monitoring in background
        self._start_health_monitoring()
        
        return True
    
    def _start_health_monitoring(self):
        """Start health monitoring in background thread"""
        def monitor_health():
            while self.running:
                try:
                    for app_name, config in self.apps.items():
                        try:
                            response = requests.get(config['health_url'], timeout=5)
                            if response.status_code == 200:
                                logger.debug(f"{config['name']} is healthy")
                            else:
                                logger.warning(f"{config['name']} returned status {response.status_code}")
                        except requests.exceptions.RequestException as e:
                            logger.warning(f"{config['name']} health check failed: {str(e)}")
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {str(e)}")
                    time.sleep(30)
        
        health_thread = threading.Thread(target=monitor_health, daemon=True)
        health_thread.start()
        logger.info("Health monitoring started")
    
    def wait_for_applications(self, timeout: int = 120) -> bool:
        """Wait for applications to be ready"""
        logger.info(f"Waiting for applications to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        ready_apps = set()
        
        while time.time() - start_time < timeout:
            for app_name, config in self.apps.items():
                if app_name in ready_apps:
                    continue
                
                try:
                    response = requests.get(config['health_url'], timeout=5)
                    if response.status_code == 200:
                        ready_apps.add(app_name)
                        logger.info(f"✅ {config['name']} is ready")
                except requests.exceptions.RequestException:
                    pass
            
            # Check if all apps are ready
            if len(ready_apps) == len(self.apps):
                logger.info("🎉 All applications are ready!")
                return True
            
            time.sleep(2)
        
        logger.error("❌ Application startup timeout")
        return False
    
    def stop_application(self, process: subprocess.Popen, app_name: str):
        """Stop a specific application"""
        logger.info(f"Stopping {app_name}...")
        
        try:
            # Send SIGTERM
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.info(f"{app_name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                process.kill()
                logger.info(f"{app_name} force killed")
                
        except Exception as e:
            logger.error(f"Error stopping {app_name}: {str(e)}")
    
    def stop_all_applications(self):
        """Stop all running applications"""
        logger.info("Stopping all applications...")
        
        self.running = False
        
        for process in self.processes:
            if process.poll() is None:  # Process is still running
                self.stop_application(process, "Unknown Application")
        
        self.processes.clear()
        logger.info("All applications stopped")
    
    def show_status(self):
        """Show status of all applications"""
        logger.info("Application Status:")
        logger.info("=" * 50)
        
        for app_name, config in self.apps.items():
            try:
                response = requests.get(config['health_url'], timeout=5)
                status = "✅ Running" if response.status_code == 200 else "❌ Error"
                logger.info(f"{config['name']}: {status} (Port: {config['port']})")
            except requests.exceptions.RequestException:
                logger.info(f"{config['name']}: ❌ Not responding (Port: {config['port']})")
        
        logger.info("=" * 50)
    
    def show_logs(self, app_name: str = None, follow: bool = False):
        """Show logs for applications"""
        if app_name and app_name not in self.apps:
            logger.error(f"Unknown application: {app_name}")
            return
        
        if app_name:
            # Show logs for specific application
            self._show_app_logs(app_name, follow)
        else:
            # Show logs for all applications
            for app_name in self.apps.keys():
                self._show_app_logs(app_name, follow)
    
    def _show_app_logs(self, app_name: str, follow: bool = False):
        """Show logs for a specific application"""
        app_config = self.apps[app_name]
        log_file = self.project_root / f"{app_name}.log"
        
        if not log_file.exists():
            logger.warning(f"No log file found for {app_name}")
            return
        
        logger.info(f"Showing logs for {app_name}:")
        try:
            with open(log_file, 'r') as f:
                if follow:
                    # Follow logs in real-time
                    import time
                    f.seek(0, 2)  # Go to end of file
                    while True:
                        line = f.readline()
                        if line:
                            print(line.rstrip())
                        else:
                            time.sleep(0.1)
                else:
                    # Show all logs
                    content = f.read()
                    print(content)
        except KeyboardInterrupt:
            logger.info("Log viewing interrupted")
        except Exception as e:
            logger.error(f"Error reading logs: {str(e)}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if app_manager:
        app_manager.stop_all_applications()
    sys.exit(0)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Medical Imaging Diagnosis Agent Applications")
    parser.add_argument("--action", choices=["start", "stop", "status", "logs"], 
                       default="start", help="Action to perform")
    parser.add_argument("--app", choices=["streamlit", "api"], 
                       help="Specific application to operate on")
    parser.add_argument("--follow", action="store_true", 
                       help="Follow logs in real-time")
    parser.add_argument("--timeout", type=int, default=120, 
                       help="Startup timeout in seconds")
    
    args = parser.parse_args()
    
    global app_manager
    app_manager = ApplicationManager()
    
    try:
        if args.action == "start":
            # Start applications
            if app_manager.start_all_applications():
                # Wait for them to be ready
                if app_manager.wait_for_applications(args.timeout):
                    logger.info("🚀 All applications are running!")
                    logger.info("🌐 Streamlit Frontend: http://localhost:8501")
                    logger.info("🔌 FastAPI Backend: http://localhost:8000")
                    logger.info("📚 API Documentation: http://localhost:8000/docs")
                    logger.info("📊 Health Check: http://localhost:8000/health")
                    logger.info("")
                    logger.info("Press Ctrl+C to stop all applications")
                    
                    # Keep running until interrupted
                    try:
                        while app_manager.running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Shutdown requested by user")
                else:
                    logger.error("Applications failed to start properly")
                    sys.exit(1)
            else:
                logger.error("Failed to start applications")
                sys.exit(1)
                
        elif args.action == "stop":
            app_manager.stop_all_applications()
            
        elif args.action == "status":
            app_manager.show_status()
            
        elif args.action == "logs":
            app_manager.show_logs(args.app, args.follow)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if app_manager:
            app_manager.stop_all_applications()

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app_manager = None
    main()
