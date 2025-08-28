"""
Configuration file for Advanced Medical Imaging Diagnosis Agent
"""
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for the medical imaging diagnosis system"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    UPLOADS_DIR = BASE_DIR / "uploads"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Create directories if they don't exist
    for dir_path in [UPLOADS_DIR, MODELS_DIR, REPORTS_DIR, CACHE_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
    
    # PubMed Configuration
    PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
    PUBMED_TOOL = "AdvancedMedicalImagingAgent"
    PUBMED_MAX_RESULTS = int(os.getenv("PUBMED_MAX_RESULTS", "10"))
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///medical_imaging.db")
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/")
    
    # Model Configuration
    MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
    MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.7"))
    MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", "4"))
    
    # Image Processing
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
    SUPPORTED_FORMATS = ['.dcm', '.nii', '.nii.gz', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    DICOM_TAGS = [
        'PatientName', 'PatientID', 'StudyDate', 'Modality', 
        'BodyPartExamined', 'ImageComments'
    ]
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Streamlit Configuration
    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / "logs" / "app.log"
    
    # Monitoring
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
    
    # Model Training
    TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", str(BASE_DIR / "training_data"))
    VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.2"))
    EPOCHS = int(os.getenv("EPOCHS", "100"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    
    # XAI Configuration
    GRAD_CAM_LAYER = os.getenv("GRAD_CAM_LAYER", "layer4")
    HEATMAP_ALPHA = float(os.getenv("HEATMAP_ALPHA", "0.6"))
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get the path for a specific model"""
        return cls.MODELS_DIR / model_name
    
    @classmethod
    def get_upload_path(cls, filename: str) -> Path:
        """Get the upload path for a file"""
        return cls.UPLOADS_DIR / filename
    
    @classmethod
    def get_report_path(cls, report_id: str) -> Path:
        """Get the report path for a specific report"""
        return cls.REPORTS_DIR / f"{report_id}.pdf"
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = {}
        
        if not cls.OPENAI_API_KEY:
            issues["OPENAI_API_KEY"] = "OpenAI API key is required"
        
        if not cls.PUBMED_EMAIL:
            issues["PUBMED_EMAIL"] = "PubMed email is required for literature search"
        
        if not Path(cls.TRAINING_DATA_DIR).exists():
            issues["TRAINING_DATA_DIR"] = f"Training data directory does not exist: {cls.TRAINING_DATA_DIR}"
        
        return issues

# Global configuration instance
config = Config()
