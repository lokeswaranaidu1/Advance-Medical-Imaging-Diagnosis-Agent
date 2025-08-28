# 🚀 Advanced Medical Imaging Diagnosis Agent - Deployment Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)
8. [Scaling and Performance](#scaling-and-performance)

## 🔧 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: Minimum 20GB free space (50GB+ recommended)
- **CPU**: 4+ cores recommended for AI processing

### Software Requirements
- **Docker**: 20.10+ with Docker Compose
- **Git**: For version control
- **Python**: 3.9+ with pip
- **Virtual Environment**: Recommended (venv or conda)

### Network Requirements
- **Ports**: 8000 (API), 8501 (Streamlit), 6379 (Redis), 27017 (MongoDB)
- **Internet**: Required for OpenAI API and PubMed access
- **Firewall**: Configure to allow required ports

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd Advanced-Medical-Imaging-Diagnosis-Agent
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_api.txt
```

### 3. Configure Environment
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

### 4. Run Applications
```bash
# Terminal 1: Start FastAPI
python start_api.py

# Terminal 2: Start Streamlit
python start_streamlit.py
```

### 5. Access Applications
- **Streamlit App**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### 1. Build and Deploy
```bash
# Build all services
docker-compose build --no-cache

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Service Management
```bash
# Stop services
docker-compose down

# Restart specific service
docker-compose restart api

# Scale services
docker-compose up -d --scale api=3

# View resource usage
docker stats
```

### 3. Automated Deployment
```bash
# Full deployment with health checks
python deploy.py --action deploy

# Check deployment status
python deploy.py --action status

# Stop all services
python deploy.py --action stop

# Clean up completely
python deploy.py --action cleanup
```

## 🏭 Production Deployment

### 1. Production Environment Setup
```bash
# Set production environment
export ENVIRONMENT=production
export DEBUG=false

# Use production Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### 2. SSL/HTTPS Configuration
```bash
# Generate SSL certificates
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes

# Configure Nginx with SSL
cp nginx.ssl.conf nginx.conf
docker-compose restart nginx
```

### 3. Load Balancer Setup
```bash
# Example Nginx load balancer configuration
upstream api_backend {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

upstream streamlit_backend {
    server streamlit1:8501;
    server streamlit2:8501;
}
```

### 4. Database Scaling
```bash
# MongoDB replica set
docker-compose -f docker-compose.mongodb-cluster.yml up -d

# Redis cluster
docker-compose -f docker-compose.redis-cluster.yml up -d
```

## ⚙️ Environment Configuration

### Required Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000

# PubMed Configuration
PUBMED_EMAIL=your_email@domain.com
PUBMED_TOOL=AdvancedMedicalImagingAgent

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/medical_imaging
MONGODB_URL=mongodb://admin:password@localhost:27017/
REDIS_URL=redis://localhost:6379/

# Monitoring
ENABLE_MONITORING=true
METRICS_PORT=8000
```

### Environment-Specific Configs
```bash
# Development
cp env_template.txt .env.dev
export ENV_FILE=.env.dev

# Staging
cp env_template.txt .env.staging
export ENV_FILE=.env.staging

# Production
cp env_template.txt .env.prod
export ENV_FILE=.env.prod
```

## 📊 Monitoring Setup

### 1. Enable Monitoring
```bash
# Set environment variable
export ENABLE_MONITORING=true

# Start monitoring
python -c "from monitoring import system_monitor; system_monitor.start_monitoring()"
```

### 2. Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'medical-imaging-agent'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 3. Grafana Dashboard
```bash
# Start Grafana
docker run -d -p 3000:3000 --name grafana grafana/grafana

# Access Grafana
# URL: http://localhost:3000
# Default credentials: admin/admin
```

### 4. Health Check Endpoints
```bash
# System health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# Service status
curl http://localhost:8000/health/services
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000
lsof -i :8000

# Kill process using port
sudo kill -9 <PID>

# Use different ports
export STREAMLIT_SERVER_PORT=8502
export METRICS_PORT=8001
```

#### 2. Docker Issues
```bash
# Check Docker status
docker info
docker-compose --version

# Restart Docker service
sudo systemctl restart docker

# Clean up Docker
docker system prune -a
docker volume prune
```

#### 3. Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Increase Docker memory limit
# Edit Docker Desktop settings or /etc/docker/daemon.json
{
  "memory": "8g",
  "swap": "2g"
}
```

#### 4. Database Connection Issues
```bash
# Test database connectivity
python -c "from database import db_manager; print(db_manager.test_connection())"

# Check database logs
docker-compose logs mongodb
docker-compose logs redis
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python start_api.py --verbose
streamlit run app.py --logger.level debug
```

### Log Analysis
```bash
# View application logs
tail -f logs/app.log
tail -f logs/api.log

# Search for errors
grep -i error logs/*.log
grep -i exception logs/*.log

# Monitor real-time logs
docker-compose logs -f --tail=100
```

## 📈 Scaling and Performance

### 1. Horizontal Scaling
```bash
# Scale API services
docker-compose up -d --scale api=5

# Scale Streamlit services
docker-compose up -d --scale streamlit=3

# Load balancer configuration
# Update nginx.conf with multiple upstream servers
```

### 2. Performance Optimization
```bash
# Enable caching
export ENABLE_CACHING=true
export REDIS_CACHE_TTL=3600

# Optimize image processing
export IMAGE_PROCESSING_WORKERS=4
export BATCH_SIZE=8

# Database optimization
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=30
```

### 3. Resource Monitoring
```bash
# Monitor system resources
htop
iotop
nethogs

# Monitor Docker resources
docker stats --no-stream
docker system df

# Monitor application performance
curl http://localhost:8000/metrics
```

### 4. Backup and Recovery
```bash
# Database backup
docker exec mongodb mongodump --out /backup
docker cp mongodb:/backup ./backup

# Volume backup
docker run --rm -v medical_imaging_mongodb_data:/data -v $(pwd):/backup alpine tar czf /backup/mongodb_backup.tar.gz -C /data .

# Restore from backup
docker exec -i mongodb mongorestore --archive < backup/mongodb_backup.archive
```

## 🔒 Security Considerations

### 1. Network Security
```bash
# Firewall configuration
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# VPN setup for remote access
# Configure OpenVPN or WireGuard
```

### 2. Authentication
```bash
# Enable JWT authentication
export JWT_ENABLED=true
export JWT_ALGORITHM=HS256
export ACCESS_TOKEN_EXPIRE_MINUTES=30

# API key authentication
export API_KEY_AUTH_ENABLED=true
export API_KEY_HEADER=X-API-Key
```

### 3. Data Encryption
```bash
# Enable encryption at rest
export ENCRYPTION_ENABLED=true
export ENCRYPTION_KEY=your-encryption-key

# SSL/TLS configuration
export SSL_ENABLED=true
export SSL_CERT_PATH=/path/to/cert.pem
export SSL_KEY_PATH=/path/to/key.pem
```

## 📚 Additional Resources

### Documentation
- [Project README](README.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [API Documentation](http://localhost:8000/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Support
- Check logs for detailed error information
- Review configuration settings
- Verify system requirements
- Test individual components
- Monitor system resources

### Community
- GitHub Issues for bug reports
- GitHub Discussions for questions
- Contributing guidelines
- Code of conduct

---

## 🎯 Deployment Checklist

### Pre-Deployment
- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Environment configured
- [ ] Ports available
- [ ] Docker running
- [ ] API keys configured

### Deployment
- [ ] Services built successfully
- [ ] Services started
- [ ] Health checks passed
- [ ] Monitoring enabled
- [ ] Logs accessible
- [ ] Performance acceptable

### Post-Deployment
- [ ] Applications accessible
- [ ] API endpoints working
- [ ] Database connections stable
- [ ] Monitoring metrics visible
- [ ] Backup procedures tested
- [ ] Documentation updated

---

*This deployment guide provides comprehensive instructions for deploying the Advanced Medical Imaging Diagnosis Agent in various environments.*
