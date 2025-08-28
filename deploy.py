#!/usr/bin/env python3
"""
Advanced Medical Imaging Diagnosis Agent - Deployment Script
Automates deployment, monitoring, and management of the entire system
"""

import os
import sys
import subprocess
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import docker
from docker.errors import DockerException
import yaml
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalImagingDeployer:
    """Deployment manager for Medical Imaging Diagnosis Agent"""
    
    def __init__(self, config_path: str = "deployment_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.docker_client = self.init_docker_client()
        self.project_root = Path(__file__).parent
        
    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                return {
                    "environment": "production",
                    "services": ["api", "streamlit", "database", "redis", "nginx"],
                    "monitoring": ["prometheus", "grafana"],
                    "ports": {
                        "api": 8000,
                        "streamlit": 8501,
                        "database": 5432,
                        "redis": 6379,
                        "nginx": 80,
                        "prometheus": 9090,
                        "grafana": 3000
                    },
                    "health_check_timeout": 300,
                    "backup_enabled": True,
                    "ssl_enabled": False
                }
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def init_docker_client(self) -> Optional[docker.DockerClient]:
        """Initialize Docker client"""
        try:
            return docker.from_env()
        except DockerException as e:
            logger.error(f"Docker not available: {str(e)}")
            return None
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")
        
        # Check Docker
        if not self.docker_client:
            logger.error("Docker is not available")
            return False
        
        # Check Docker Compose
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                logger.error("Docker Compose is not available")
                return False
        except FileNotFoundError:
            logger.error("Docker Compose is not available")
            return False
        
        # Check required files
        required_files = [
            "docker-compose.yml",
            "Dockerfile",
            "requirements.txt",
            "requirements_api.txt"
        ]
        
        for file in required_files:
            if not (self.project_root / file).exists():
                logger.error(f"Required file not found: {file}")
                return False
        
        # Check environment variables
        required_env_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            logger.warning("Some features may not work properly")
        
        logger.info("Prerequisites check completed")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating necessary directories...")
        
        directories = [
            "uploads",
            "models", 
            "reports",
            "cache",
            "logs",
            "logs/nginx",
            "database",
            "nginx",
            "nginx/ssl",
            "monitoring",
            "monitoring/grafana",
            "monitoring/grafana/dashboards",
            "monitoring/grafana/datasources",
            "training_data",
            "backups"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def generate_environment_file(self):
        """Generate .env file from template"""
        logger.info("Generating environment file...")
        
        env_template = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if env_template.exists() and not env_file.exists():
            shutil.copy(env_template, env_file)
            logger.info("Generated .env file from template")
        elif not env_file.exists():
            # Create basic .env file
            env_content = """# Medical Imaging Diagnosis Agent Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
PUBMED_API_KEY=your_pubmed_api_key_here
DATABASE_URL=postgresql://medical_user:medical_password@database:5432/medical_imaging
REDIS_URL=redis://redis:6379
DB_USER=medical_user
DB_PASSWORD=medical_password
GRAFANA_PASSWORD=admin
LOG_LEVEL=INFO
ENVIRONMENT=production
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info("Created basic .env file")
    
    def build_images(self) -> bool:
        """Build Docker images"""
        logger.info("Building Docker images...")
        
        try:
            # Build with docker-compose
            result = subprocess.run(
                ["docker-compose", "build", "--no-cache"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return False
            
            logger.info("Docker images built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building images: {str(e)}")
            return False
    
    def start_services(self, services: Optional[List[str]] = None) -> bool:
        """Start specified services"""
        if services is None:
            services = self.config.get("services", [])
        
        logger.info(f"Starting services: {services}")
        
        try:
            # Start services with docker-compose
            cmd = ["docker-compose", "up", "-d"] + services
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Service start failed: {result.stderr}")
                return False
            
            logger.info("Services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting services: {str(e)}")
            return False
    
    def wait_for_health_checks(self, timeout: int = 300) -> bool:
        """Wait for all services to be healthy"""
        logger.info("Waiting for health checks...")
        
        start_time = time.time()
        services_healthy = set()
        
        while time.time() - start_time < timeout:
            try:
                # Check API health
                if "api" not in services_healthy:
                    try:
                        response = requests.get("http://localhost:8000/health", timeout=5)
                        if response.status_code == 200:
                            services_healthy.add("api")
                            logger.info("API service is healthy")
                    except:
                        pass
                
                # Check Streamlit health
                if "streamlit" not in services_healthy:
                    try:
                        response = requests.get("http://localhost:8501", timeout=5)
                        if response.status_code == 200:
                            services_healthy.add("streamlit")
                            logger.info("Streamlit service is healthy")
                    except:
                        pass
                
                # Check if all required services are healthy
                required_services = set(self.config.get("services", []))
                if required_services.issubset(services_healthy):
                    logger.info("All required services are healthy")
                    return True
                
                time.sleep(5)
                
            except Exception as e:
                logger.warning(f"Health check error: {str(e)}")
                time.sleep(5)
        
        logger.error("Health check timeout")
        return False
    
    def check_service_status(self) -> Dict[str, Any]:
        """Check status of all services"""
        logger.info("Checking service status...")
        
        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Service status:\n" + result.stdout)
                return {"status": "success", "output": result.stdout}
            else:
                logger.error(f"Status check failed: {result.stderr}")
                return {"status": "error", "error": result.stderr}
                
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def stop_services(self, services: Optional[List[str]] = None) -> bool:
        """Stop specified services"""
        if services is None:
            services = self.config.get("services", [])
        
        logger.info(f"Stopping services: {services}")
        
        try:
            cmd = ["docker-compose", "stop"] + services
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Service stop failed: {result.stderr}")
                return False
            
            logger.info("Services stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping services: {str(e)}")
            return False
    
    def restart_services(self, services: Optional[List[str]] = None) -> bool:
        """Restart specified services"""
        if services is None:
            services = self.config.get("services", [])
        
        logger.info(f"Restarting services: {services}")
        
        try:
            # Stop services
            if not self.stop_services(services):
                return False
            
            # Wait a moment
            time.sleep(5)
            
            # Start services
            if not self.start_services(services):
                return False
            
            logger.info("Services restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error restarting services: {str(e)}")
            return False
    
    def view_logs(self, service: str = None, follow: bool = False):
        """View logs for services"""
        try:
            cmd = ["docker-compose", "logs"]
            
            if follow:
                cmd.append("-f")
            
            if service:
                cmd.append(service)
            
            # Run logs command
            subprocess.run(cmd, cwd=self.project_root)
            
        except Exception as e:
            logger.error(f"Error viewing logs: {str(e)}")
    
    def backup_data(self) -> bool:
        """Create backup of important data"""
        if not self.config.get("backup_enabled", True):
            logger.info("Backups are disabled")
            return True
        
        logger.info("Creating backup...")
        
        try:
            backup_dir = self.project_root / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = backup_dir / backup_name
            
            # Create backup archive
            import tarfile
            
            with tarfile.open(backup_path.with_suffix('.tar.gz'), 'w:gz') as tar:
                # Add important directories
                important_dirs = ['uploads', 'models', 'reports', 'database']
                for dir_name in important_dirs:
                    dir_path = self.project_root / dir_name
                    if dir_path.exists():
                        tar.add(dir_path, arcname=dir_name)
                
                # Add configuration files
                config_files = ['.env', 'docker-compose.yml', 'config.py']
                for file_name in config_files:
                    file_path = self.project_root / file_name
                    if file_path.exists():
                        tar.add(file_path, arcname=file_name)
            
            logger.info(f"Backup created: {backup_path.with_suffix('.tar.gz')}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return False
    
    def cleanup(self) -> bool:
        """Clean up containers and images"""
        logger.info("Cleaning up...")
        
        try:
            # Stop and remove containers
            subprocess.run(
                ["docker-compose", "down", "-v"],
                cwd=self.project_root,
                capture_output=True
            )
            
            # Remove images
            subprocess.run(
                ["docker-compose", "down", "--rmi", "all"],
                cwd=self.project_root,
                capture_output=True
            )
            
            logger.info("Cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return False
    
    def deploy(self) -> bool:
        """Complete deployment process"""
        logger.info("Starting deployment...")
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                return False
            
            # Create directories
            self.create_directories()
            
            # Generate environment file
            self.generate_environment_file()
            
            # Build images
            if not self.build_images():
                return False
            
            # Start services
            if not self.start_services():
                return False
            
            # Wait for health checks
            if not self.wait_for_health_checks():
                return False
            
            # Create backup
            self.backup_data()
            
            logger.info("Deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return False

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Medical Imaging Diagnosis Agent Deployer")
    parser.add_argument("--action", choices=[
        "deploy", "start", "stop", "restart", "status", "logs", "backup", "cleanup"
    ], default="deploy", help="Action to perform")
    
    parser.add_argument("--services", nargs="+", help="Specific services to operate on")
    parser.add_argument("--follow", action="store_true", help="Follow logs")
    parser.add_argument("--config", default="deployment_config.json", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = MedicalImagingDeployer(args.config)
    
    try:
        if args.action == "deploy":
            success = deployer.deploy()
            sys.exit(0 if success else 1)
            
        elif args.action == "start":
            success = deployer.start_services(args.services)
            sys.exit(0 if success else 1)
            
        elif args.action == "stop":
            success = deployer.stop_services(args.services)
            sys.exit(0 if success else 1)
            
        elif args.action == "restart":
            success = deployer.restart_services(args.services)
            sys.exit(0 if success else 1)
            
        elif args.action == "status":
            result = deployer.check_service_status()
            print(json.dumps(result, indent=2))
            
        elif args.action == "logs":
            service = args.services[0] if args.services else None
            deployer.view_logs(service, args.follow)
            
        elif args.action == "backup":
            success = deployer.backup_data()
            sys.exit(0 if success else 1)
            
        elif args.action == "cleanup":
            success = deployer.cleanup()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
