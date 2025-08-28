"""
Database Models and Management for Medical Imaging Diagnosis System
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from datetime import datetime
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(config.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    """User model for authentication and collaboration"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), default="doctor")  # doctor, radiologist, admin
    specialty = Column(String(100))
    institution = Column(String(200))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    cases = relationship("MedicalCase", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")

class MedicalCase(Base):
    """Medical case model for storing diagnosis sessions"""
    __tablename__ = "medical_cases"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_id = Column(String(100))  # Anonymized patient identifier
    case_title = Column(String(200), nullable=False)
    case_description = Column(Text)
    modality = Column(String(50))  # CT, MRI, X-ray, etc.
    body_part = Column(String(100))
    urgency_level = Column(String(20), default="routine")  # routine, urgent, emergent
    status = Column(String(20), default="pending")  # pending, in_progress, completed, archived
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="cases")
    images = relationship("MedicalImage", back_populates="case")
    diagnoses = relationship("Diagnosis", back_populates="case")
    chat_messages = relationship("ChatMessage", back_populates="case")

class MedicalImage(Base):
    """Medical image model for storing image data and metadata"""
    __tablename__ = "medical_images"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("medical_cases.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    original_filename = Column(String(200), nullable=False)
    image_format = Column(String(20))  # DICOM, NIfTI, JPG, PNG, etc.
    image_size = Column(String(50))  # Dimensions
    file_size = Column(Integer)  # File size in bytes
    metadata = Column(JSON)  # DICOM tags, etc.
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    case = relationship("MedicalCase", back_populates="images")
    diagnoses = relationship("Diagnosis", back_populates="image")

class Diagnosis(Base):
    """Diagnosis model for storing AI and human diagnoses"""
    __tablename__ = "diagnoses"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("medical_cases.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("medical_images.id"))
    diagnosis_text = Column(Text, nullable=False)
    confidence_score = Column(Float)
    diagnosis_type = Column(String(20))  # ai, human, consensus
    urgency_level = Column(String(20))
    key_findings = Column(JSON)
    recommendations = Column(JSON)
    heatmap_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    case = relationship("MedicalCase", back_populates="diagnoses")
    image = relationship("MedicalImage", back_populates="diagnoses")

class ChatMessage(Base):
    """Chat message model for doctor collaboration"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("medical_cases.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message_text = Column(Text, nullable=False)
    message_type = Column(String(20), default="text")  # text, image, file
    attachment_path = Column(String(500))
    is_system_message = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    case = relationship("MedicalCase", back_populates="chat_messages")
    user = relationship("User", back_populates="chat_messages")

class LiteratureReference(Base):
    """Literature reference model for PubMed articles"""
    __tablename__ = "literature_references"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("medical_cases.id"), nullable=False)
    pmid = Column(String(20), unique=True, index=True)
    title = Column(String(500), nullable=False)
    abstract = Column(Text)
    authors = Column(JSON)
    journal = Column(String(200))
    publication_date = Column(String(50))
    url = Column(String(500))
    relevance_score = Column(Float)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    case = relationship("MedicalCase")

class DatabaseManager:
    """Database management and operations"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def create_user(self, user_data: Dict[str, Any]) -> Optional[User]:
        """Create a new user"""
        try:
            with self.get_session() as session:
                user = User(**user_data)
                session.add(user)
                session.commit()
                session.refresh(user)
                logger.info(f"User created: {user.username}")
                return user
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with self.get_session() as session:
                return session.query(User).filter(User.username == username).first()
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    def create_case(self, case_data: Dict[str, Any]) -> Optional[MedicalCase]:
        """Create a new medical case"""
        try:
            with self.get_session() as session:
                case = MedicalCase(**case_data)
                session.add(case)
                session.commit()
                session.refresh(case)
                logger.info(f"Case created: {case.case_id}")
                return case
        except Exception as e:
            logger.error(f"Error creating case: {str(e)}")
            return None
    
    def get_case_by_id(self, case_id: str) -> Optional[MedicalCase]:
        """Get case by case ID"""
        try:
            with self.get_session() as session:
                return session.query(MedicalCase).filter(MedicalCase.case_id == case_id).first()
        except Exception as e:
            logger.error(f"Error getting case: {str(e)}")
            return None
    
    def add_image_to_case(self, case_id: int, image_data: Dict[str, Any]) -> Optional[MedicalImage]:
        """Add image to a case"""
        try:
            with self.get_session() as session:
                image = MedicalImage(case_id=case_id, **image_data)
                session.add(image)
                session.commit()
                session.refresh(image)
                logger.info(f"Image added to case {case_id}")
                return image
        except Exception as e:
            logger.error(f"Error adding image to case: {str(e)}")
            return None
    
    def add_diagnosis(self, diagnosis_data: Dict[str, Any]) -> Optional[Diagnosis]:
        """Add diagnosis to a case"""
        try:
            with self.get_session() as session:
                diagnosis = Diagnosis(**diagnosis_data)
                session.add(diagnosis)
                session.commit()
                session.refresh(diagnosis)
                logger.info(f"Diagnosis added to case {diagnosis_data.get('case_id')}")
                return diagnosis
        except Exception as e:
            logger.error(f"Error adding diagnosis: {str(e)}")
            return None
    
    def add_chat_message(self, message_data: Dict[str, Any]) -> Optional[ChatMessage]:
        """Add chat message to a case"""
        try:
            with self.get_session() as session:
                message = ChatMessage(**message_data)
                session.add(message)
                session.commit()
                session.refresh(message)
                logger.info(f"Chat message added to case {message_data.get('case_id')}")
                return message
        except Exception as e:
            logger.error(f"Error adding chat message: {str(e)}")
            return None
    
    def get_case_chat_history(self, case_id: int) -> List[ChatMessage]:
        """Get chat history for a case"""
        try:
            with self.get_session() as session:
                messages = session.query(ChatMessage).filter(
                    ChatMessage.case_id == case_id
                ).order_by(ChatMessage.created_at).all()
                return messages
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
    
    def search_cases(self, search_term: str, user_id: int = None) -> List[MedicalCase]:
        """Search cases by various criteria"""
        try:
            with self.get_session() as session:
                query = session.query(MedicalCase)
                
                if user_id:
                    query = query.filter(MedicalCase.user_id == user_id)
                
                if search_term:
                    query = query.filter(
                        MedicalCase.case_title.contains(search_term) |
                        MedicalCase.case_description.contains(search_term) |
                        MedicalCase.patient_id.contains(search_term)
                    )
                
                return query.order_by(MedicalCase.created_at.desc()).all()
        except Exception as e:
            logger.error(f"Error searching cases: {str(e)}")
            return []
    
    def update_case_status(self, case_id: str, status: str) -> bool:
        """Update case status"""
        try:
            with self.get_session() as session:
                case = session.query(MedicalCase).filter(MedicalCase.case_id == case_id).first()
                if case:
                    case.status = status
                    case.updated_at = datetime.now()
                    session.commit()
                    logger.info(f"Case {case_id} status updated to {status}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating case status: {str(e)}")
            return False
    
    def get_user_cases(self, user_id: int) -> List[MedicalCase]:
        """Get all cases for a user"""
        try:
            with self.get_session() as session:
                return session.query(MedicalCase).filter(
                    MedicalCase.user_id == user_id
                ).order_by(MedicalCase.created_at.desc()).all()
        except Exception as e:
            logger.error(f"Error getting user cases: {str(e)}")
            return []
    
    def delete_case(self, case_id: str) -> bool:
        """Delete a case and all associated data"""
        try:
            with self.get_session() as session:
                case = session.query(MedicalCase).filter(MedicalCase.case_id == case_id).first()
                if case:
                    session.delete(case)
                    session.commit()
                    logger.info(f"Case {case_id} deleted")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting case: {str(e)}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

# Create tables on import
try:
    db_manager.create_tables()
except Exception as e:
    logger.warning(f"Could not create database tables: {str(e)}")
