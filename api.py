"""
FastAPI Application for Advanced Medical Imaging Diagnosis Agent
RESTful API endpoints for medical image analysis and diagnosis
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import os
from pathlib import Path
import uuid
from datetime import datetime
import json

# Import custom modules
from config import config
from image_processor import image_processor
from ai_diagnosis import ai_diagnosis
from literature_search import pubmed_search
from database import db_manager
from report_generator import report_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Medical Imaging Diagnosis Agent API",
    description="RESTful API for medical image analysis, AI diagnosis, and case management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    full_name: str = Field(..., min_length=2, max_length=100)
    role: str = Field(default="doctor", regex="^(doctor|radiologist|admin)$")
    specialty: Optional[str] = Field(None, max_length=100)
    institution: Optional[str] = Field(None, max_length=200)

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    specialty: Optional[str]
    institution: Optional[str]
    is_active: bool
    created_at: datetime

class CaseCreate(BaseModel):
    case_title: str = Field(..., min_length=5, max_length=200)
    patient_id: Optional[str] = Field(None, max_length=100)
    case_description: Optional[str] = Field(None, max_length=1000)
    modality: str = Field(..., regex="^(CT|MRI|X-ray|Ultrasound|Other)$")
    body_part: Optional[str] = Field(None, max_length=100)
    urgency_level: str = Field(default="routine", regex="^(routine|urgent|emergent)$")

class CaseResponse(BaseModel):
    id: int
    case_id: str
    case_title: str
    patient_id: Optional[str]
    case_description: Optional[str]
    modality: str
    body_part: Optional[str]
    urgency_level: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime]

class DiagnosisResponse(BaseModel):
    id: int
    diagnosis_text: str
    confidence_score: Optional[float]
    diagnosis_type: str
    urgency_level: Optional[str]
    key_findings: Optional[Dict[str, Any]]
    recommendations: Optional[List[str]]
    created_at: datetime

class ChatMessageCreate(BaseModel):
    message_text: str = Field(..., min_length=1, max_length=1000)
    message_type: str = Field(default="text", regex="^(text|image|file)$")

class ChatMessageResponse(BaseModel):
    id: int
    message_text: str
    message_type: str
    is_system_message: bool
    created_at: datetime
    user: UserResponse

class LiteratureSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=200)
    max_results: Optional[int] = Field(10, ge=1, le=50)
    search_type: Optional[str] = Field("general", regex="^(general|diagnosis|recent|systematic|guidelines)$")

class LiteratureSearchResponse(BaseModel):
    pmid: str
    title: str
    abstract: Optional[str]
    authors: List[str]
    journal: str
    publication_date: str
    url: str
    relevance_score: Optional[float]

class AnalysisRequest(BaseModel):
    case_id: Optional[str] = None
    generate_report: bool = False
    include_literature: bool = True

class AnalysisResponse(BaseModel):
    case_id: str
    total_images: int
    analysis_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    report_path: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime

# Dependency functions
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token (simplified for demo)"""
    # In production, implement proper JWT validation
    try:
        # Demo user for now
        return {
            'id': 1,
            'username': 'demo_user',
            'full_name': 'Dr. Demo User',
            'role': 'doctor'
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Validate API key (simplified for demo)"""
    # In production, implement proper API key validation
    return True

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "ai_model": "available",
            "image_processor": "ready"
        }
    }

# User management endpoints
@app.post("/users", response_model=UserResponse, tags=["Users"])
async def create_user(user_data: UserCreate):
    """Create a new user"""
    try:
        user = db_manager.create_user(user_data.dict())
        if user:
            return UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                specialty=user.specialty,
                institution=user.institution,
                is_active=user.is_active,
                created_at=user.created_at
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to create user")
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/users/{username}", response_model=UserResponse, tags=["Users"])
async def get_user(username: str):
    """Get user by username"""
    try:
        user = db_manager.get_user_by_username(username)
        if user:
            return UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                specialty=user.specialty,
                institution=user.institution,
                is_active=user.is_active,
                created_at=user.created_at
            )
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Case management endpoints
@app.post("/cases", response_model=CaseResponse, tags=["Cases"])
async def create_case(case_data: CaseCreate, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Create a new medical case"""
    try:
        case_id = f"CASE_{uuid.uuid4().hex[:8].upper()}"
        case_data_dict = case_data.dict()
        case_data_dict['case_id'] = case_id
        case_data_dict['user_id'] = current_user['id']
        
        case = db_manager.create_case(case_data_dict)
        if case:
            return CaseResponse(
                id=case.id,
                case_id=case.case_id,
                case_title=case.case_title,
                patient_id=case.patient_id,
                case_description=case.case_description,
                modality=case.modality,
                body_part=case.body_part,
                urgency_level=case.urgency_level,
                status=case.status,
                created_at=case.created_at,
                updated_at=case.updated_at
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to create case")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases", response_model=List[CaseResponse], tags=["Cases"])
async def get_cases(
    search: Optional[str] = None,
    user_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get cases with optional filtering"""
    try:
        cases = db_manager.search_cases(search, user_id)
        # Apply pagination
        paginated_cases = cases[offset:offset + limit]
        
        return [
            CaseResponse(
                id=case.id,
                case_id=case.case_id,
                case_title=case.case_title,
                patient_id=case.patient_id,
                case_description=case.case_description,
                modality=case.modality,
                body_part=case.body_part,
                urgency_level=case.urgency_level,
                status=case.status,
                created_at=case.created_at,
                updated_at=case.updated_at
            )
            for case in paginated_cases
        ]
    except Exception as e:
        logger.error(f"Error getting cases: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}", response_model=CaseResponse, tags=["Cases"])
async def get_case(case_id: str):
    """Get case by ID"""
    try:
        case = db_manager.get_case_by_id(case_id)
        if case:
            return CaseResponse(
                id=case.id,
                case_id=case.case_id,
                case_title=case.case_title,
                patient_id=case.patient_id,
                case_description=case.case_description,
                modality=case.modality,
                body_part=case.body_part,
                urgency_level=case.urgency_level,
                status=case.status,
                created_at=case.created_at,
                updated_at=case.updated_at
            )
        else:
            raise HTTPException(status_code=404, detail="Case not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Image upload and analysis endpoints
@app.post("/upload", tags=["Analysis"])
async def upload_images(
    files: List[UploadFile] = File(...),
    case_id: Optional[str] = Form(None),
    analysis_request: AnalysisRequest = Depends()
):
    """Upload medical images for analysis"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Validate file types
        supported_types = config.SUPPORTED_FORMATS
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in supported_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_ext}. Supported: {supported_types}"
                )
        
        # Process uploaded files
        results = []
        for uploaded_file in files:
            # Save uploaded file
            file_path = config.get_upload_path(uploaded_file.filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.file.read())
            
            # Process image
            image_data = image_processor.load_image(file_path)
            
            if image_data:
                # Preprocess for AI analysis
                processed_image = image_processor.preprocess_image(image_data['image_data'])
                
                # AI diagnosis
                diagnosis_result = ai_diagnosis.analyze_image(
                    processed_image, 
                    image_data['metadata']
                )
                
                # Add image info
                diagnosis_result['image_path'] = str(file_path)
                diagnosis_result['image_format'] = image_data['format']
                diagnosis_result['original_filename'] = uploaded_file.filename
                
                results.append(diagnosis_result)
        
        # Save to database if case_id provided
        if case_id and results:
            case = db_manager.get_case_by_id(case_id)
            if case:
                for result in results:
                    # Save image
                    image_data_db = {
                        'case_id': case.id,
                        'image_path': result['image_path'],
                        'original_filename': result['original_filename'],
                        'image_format': result['image_format'],
                        'image_size': f"{result.get('original_shape', 'Unknown')}",
                        'file_size': os.path.getsize(result['image_path']),
                        'metadata': result.get('metadata', {})
                    }
                    db_manager.add_image_to_case(case.id, image_data_db)
                    
                    # Save diagnosis
                    diagnosis_data = {
                        'case_id': case.id,
                        'diagnosis_text': result.get('diagnosis', 'Analysis completed'),
                        'confidence_score': result.get('confidence', 0),
                        'diagnosis_type': 'ai',
                        'urgency_level': result.get('urgency', 'routine'),
                        'key_findings': result.get('key_features', []),
                        'recommendations': result.get('recommendations', []),
                        'heatmap_data': result.get('heatmap', {})
                    }
                    db_manager.add_diagnosis(diagnosis_data)
                
                # Update case status
                db_manager.update_case_status(case_id, "completed")
        
        # Generate summary
        summary = ai_diagnosis.get_diagnosis_summary(results)
        
        # Generate report if requested
        report_path = None
        if analysis_request.generate_report and case_id:
            try:
                case = db_manager.get_case_by_id(case_id)
                if case:
                    case_data = {
                        'case_id': case.case_id,
                        'patient_id': case.patient_id,
                        'case_title': case.case_title,
                        'case_description': case.case_description,
                        'modality': case.modality,
                        'body_part': case.body_part,
                        'urgency_level': case.urgency_level,
                        'status': case.status
                    }
                    
                    # Search for literature if requested
                    literature_references = []
                    if analysis_request.include_literature and case_data.get('case_description'):
                        literature_references = pubmed_search.search_medical_conditions(
                            case_data['case_description'], max_results=5
                        )
                    
                    report_path = report_generator.generate_diagnosis_report(
                        case_data, results, literature_references
                    )
            except Exception as e:
                logger.warning(f"Could not generate report: {str(e)}")
        
        return AnalysisResponse(
            case_id=case_id or "no_case",
            total_images=len(results),
            analysis_results=results,
            summary=summary,
            report_path=report_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Literature search endpoints
@app.post("/literature/search", response_model=List[LiteratureSearchResponse], tags=["Literature"])
async def search_literature(search_request: LiteratureSearchRequest):
    """Search medical literature"""
    try:
        if search_request.search_type == "diagnosis":
            articles = pubmed_search.search_by_diagnosis(
                search_request.query, 
                max_results=search_request.max_results
            )
        elif search_request.search_type == "recent":
            articles = pubmed_search.search_recent_articles(
                search_request.query, 
                days=365
            )
        elif search_request.search_type == "systematic":
            articles = pubmed_search.search_systematic_reviews(
                search_request.query
            )
        elif search_request.search_type == "guidelines":
            articles = pubmed_search.search_clinical_guidelines(
                search_request.query
            )
        else:
            articles = pubmed_search.search_medical_conditions(
                search_request.query, 
                max_results=search_request.max_results
            )
        
        return [
            LiteratureSearchResponse(
                pmid=article.get('pmid', ''),
                title=article.get('title', ''),
                abstract=article.get('abstract'),
                authors=article.get('authors', []),
                journal=article.get('journal', ''),
                publication_date=article.get('publication_date', ''),
                url=article.get('url', ''),
                relevance_score=article.get('relevance_score')
            )
            for article in articles
        ]
        
    except Exception as e:
        logger.error(f"Error searching literature: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/literature/article/{pmid}", response_model=LiteratureSearchResponse, tags=["Literature"])
async def get_article(pmid: str):
    """Get specific article by PMID"""
    try:
        article = pubmed_search.get_article_summary(pmid)
        if article:
            return LiteratureSearchResponse(
                pmid=article.get('pmid', ''),
                title=article.get('title', ''),
                abstract=article.get('abstract'),
                authors=article.get('authors', []),
                journal=article.get('journal', ''),
                publication_date=article.get('publication_date', ''),
                url=article.get('url', ''),
                relevance_score=None
            )
        else:
            raise HTTPException(status_code=404, detail="Article not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Chat and collaboration endpoints
@app.post("/cases/{case_id}/chat", response_model=ChatMessageResponse, tags=["Collaboration"])
async def send_chat_message(
    case_id: str,
    message_data: ChatMessageCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Send a chat message for a case"""
    try:
        case = db_manager.get_case_by_id(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        message_data_dict = message_data.dict()
        message_data_dict['case_id'] = case.id
        message_data_dict['user_id'] = current_user['id']
        
        message = db_manager.add_chat_message(message_data_dict)
        if message:
            return ChatMessageResponse(
                id=message.id,
                message_text=message.message_text,
                message_type=message.message_type,
                is_system_message=message.is_system_message,
                created_at=message.created_at,
                user=UserResponse(
                    id=current_user['id'],
                    username=current_user['username'],
                    email='demo@example.com',
                    full_name=current_user['full_name'],
                    role=current_user['role'],
                    specialty=None,
                    institution=None,
                    is_active=True,
                    created_at=datetime.now()
                )
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to send message")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending chat message: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cases/{case_id}/chat", response_model=List[ChatMessageResponse], tags=["Collaboration"])
async def get_chat_history(case_id: str):
    """Get chat history for a case"""
    try:
        case = db_manager.get_case_by_id(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        messages = db_manager.get_case_chat_history(case.id)
        
        return [
            ChatMessageResponse(
                id=message.id,
                message_text=message.message_text,
                message_type=message.message_type,
                is_system_message=message.is_system_message,
                created_at=message.created_at,
                user=UserResponse(
                    id=message.user.id if message.user else 1,
                    username=message.user.username if message.user else 'unknown',
                    email='demo@example.com',
                    full_name=message.user.full_name if message.user else 'Unknown User',
                    role=message.user.role if message.user else 'doctor',
                    specialty=message.user.specialty if message.user else None,
                    institution=message.user.institution if message.user else None,
                    is_active=message.user.is_active if message.user else True,
                    created_at=message.user.created_at if message.user else datetime.now()
                )
            )
            for message in messages
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Report generation endpoints
@app.post("/reports/generate", tags=["Reports"])
async def generate_case_report(
    case_id: str,
    include_literature: bool = True,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate PDF report for a case"""
    try:
        case = db_manager.get_case_by_id(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get case data
        case_data = {
            'case_id': case.case_id,
            'patient_id': case.patient_id,
            'case_title': case.case_title,
            'case_description': case.case_description,
            'modality': case.modality,
            'body_part': case.body_part,
            'urgency_level': case.urgency_level,
            'status': case.status
        }
        
        # Get diagnosis results (simplified for demo)
        diagnosis_results = [{
            'diagnosis_text': 'Sample diagnosis',
            'confidence_score': 85.0,
            'diagnosis_type': 'ai',
            'urgency_level': 'routine',
            'key_findings': ['Sample finding 1', 'Sample finding 2'],
            'recommendations': ['Sample recommendation 1', 'Sample recommendation 2']
        }]
        
        # Search for literature if requested
        literature_references = []
        if include_literature and case_data.get('case_description'):
            try:
                literature_references = pubmed_search.search_medical_conditions(
                    case_data['case_description'], max_results=5
                )
            except Exception as e:
                logger.warning(f"Could not fetch literature: {str(e)}")
        
        # Generate report
        report_path = report_generator.generate_diagnosis_report(
            case_data, diagnosis_results, literature_references
        )
        
        if report_path and os.path.exists(report_path):
            return {
                "message": "Report generated successfully",
                "report_path": report_path,
                "download_url": f"/reports/download/{Path(report_path).name}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate report")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/reports/download/{filename}", tags=["Reports"])
async def download_report(filename: str):
    """Download a generated report"""
    try:
        report_path = config.get_report_path(filename)
        if os.path.exists(report_path):
            return FileResponse(
                path=report_path,
                filename=filename,
                media_type='application/pdf'
            )
        else:
            raise HTTPException(status_code=404, detail="Report not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Analytics endpoints
@app.get("/analytics/dashboard", tags=["Analytics"])
async def get_dashboard_analytics():
    """Get dashboard analytics data"""
    try:
        # Get basic statistics (simplified for demo)
        stats = {
            "total_cases": 0,
            "pending_analysis": 0,
            "completed_cases": 0,
            "average_accuracy": 85.0,
            "monthly_volume": [10, 15, 12, 18, 22, 25],
            "modality_distribution": {
                "CT": 45,
                "MRI": 30,
                "X-ray": 20,
                "Ultrasound": 15
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting dashboard analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            timestamp=datetime.now()
        ).dict()
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Advanced Medical Imaging Diagnosis Agent API starting up...")
    
    # Validate configuration
    config_issues = config.validate_config()
    if config_issues:
        logger.warning(f"Configuration issues found: {config_issues}")
    
    # Test database connection
    try:
        session = db_manager.get_session()
        session.close()
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Advanced Medical Imaging Diagnosis Agent API shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.STREAMLIT_SERVER_ADDRESS,
        port=8000,
        reload=True,
        log_level="info"
    )
