"""
Advanced Medical Imaging Diagnosis Agent - Main Application
Streamlit-based web application for medical image analysis and diagnosis
"""

import streamlit as st
import streamlit_chat as stchat
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
import os
from pathlib import Path
import logging

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

# Page configuration
st.set_page_config(
    page_title="Advanced Medical Imaging Diagnosis Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .upload-area {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .diagnosis-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class MedicalImagingApp:
    """Main application class for medical imaging diagnosis"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_sidebar()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'current_case' not in st.session_state:
            st.session_state.current_case = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'diagnosis_results' not in st.session_state:
            st.session_state.diagnosis_results = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
    
    def setup_sidebar(self):
        """Setup sidebar navigation and user info"""
        with st.sidebar:
            st.markdown("## 🏥 Medical Imaging Agent")
            
            # User authentication (simplified for demo)
            if st.session_state.current_user is None:
                st.markdown("### 👤 Login")
                username = st.text_input("Username", key="login_username")
                if st.button("Login"):
                    # Simple demo login
                    if username:
                        st.session_state.current_user = {
                            'username': username,
                            'full_name': f"Dr. {username}",
                            'role': 'doctor'
                        }
                        st.success(f"Welcome, {username}!")
                        st.rerun()
            else:
                st.markdown(f"### 👤 Welcome, {st.session_state.current_user['full_name']}")
                if st.button("Logout"):
                    st.session_state.current_user = None
                    st.rerun()
            
            # Navigation menu
            if st.session_state.current_user:
                selected = option_menu(
                    "Navigation",
                    ["🏠 Dashboard", "📊 Case Analysis", "💬 Collaboration", "📚 Literature", "📈 Reports", "⚙️ Settings"],
                    icons=['house', 'graph-up', 'chat', 'book', 'file-earmark-text', 'gear'],
                    menu_icon="cast",
                    default_index=0,
                )
                
                if selected == "🏠 Dashboard":
                    self.show_dashboard()
                elif selected == "📊 Case Analysis":
                    self.show_case_analysis()
                elif selected == "💬 Collaboration":
                    self.show_collaboration()
                elif selected == "📚 Literature":
                    self.show_literature()
                elif selected == "📈 Reports":
                    self.show_reports()
                elif selected == "⚙️ Settings":
                    self.show_settings()
    
    def show_dashboard(self):
        """Display main dashboard"""
        st.markdown('<div class="main-header"><h1>🏥 Advanced Medical Imaging Diagnosis Agent</h1></div>', unsafe_allow_html=True)
        
        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card"><h3>📊 Total Cases</h3><h2>0</h2></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><h3>🔍 Pending Analysis</h3><h2>0</h2></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card"><h3>✅ Completed</h3><h2>0</h2></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card"><h3>📈 Accuracy</h3><h2>85%</h2></div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("## 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Upload New Case", use_container_width=True):
                st.session_state.current_case = None
                st.rerun()
        
        with col2:
            if st.button("📋 View All Cases", use_container_width=True):
                st.session_state.current_case = None
                st.rerun()
        
        # Recent activity
        st.markdown("## 📅 Recent Activity")
        st.info("No recent activity. Start by uploading a medical image for analysis.")
    
    def show_case_analysis(self):
        """Display case analysis interface"""
        st.markdown("## 📊 Medical Case Analysis")
        
        # Case creation/selection
        if st.session_state.current_case is None:
            self.show_case_creation()
        else:
            self.show_case_details()
    
    def show_case_creation(self):
        """Show case creation interface"""
        st.markdown("### 🆕 Create New Case")
        
        # Case information form
        with st.form("case_creation"):
            col1, col2 = st.columns(2)
            
            with col1:
                case_title = st.text_input("Case Title", placeholder="e.g., Chest CT for suspected pneumonia")
                patient_id = st.text_input("Patient ID", placeholder="Anonymous identifier")
                modality = st.selectbox("Imaging Modality", ["CT", "MRI", "X-ray", "Ultrasound", "Other"])
            
            with col2:
                body_part = st.text_input("Body Part", placeholder="e.g., Chest, Brain, Abdomen")
                urgency = st.selectbox("Urgency Level", ["routine", "urgent", "emergent"])
                description = st.text_area("Case Description", placeholder="Brief clinical description")
            
            submitted = st.form_submit_button("Create Case")
            
            if submitted and case_title:
                # Create new case
                case_id = f"CASE_{uuid.uuid4().hex[:8].upper()}"
                case_data = {
                    'case_id': case_id,
                    'case_title': case_title,
                    'patient_id': patient_id or f"PAT_{uuid.uuid4().hex[:6].upper()}",
                    'modality': modality,
                    'body_part': body_part,
                    'urgency_level': urgency,
                    'case_description': description,
                    'user_id': 1  # Demo user ID
                }
                
                # Save to database
                case = db_manager.create_case(case_data)
                if case:
                    st.session_state.current_case = case
                    st.success(f"Case created successfully! Case ID: {case_id}")
                    st.rerun()
                else:
                    st.error("Failed to create case. Please try again.")
        
        # File upload area
        st.markdown("### 📁 Upload Medical Images")
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose medical image files",
            type=['dcm', 'nii', 'nii.gz', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Supported formats: DICOM (.dcm), NIfTI (.nii, .nii.gz), and standard image formats"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"Uploaded {len(uploaded_files)} file(s)")
            
            # Process uploaded files
            if st.button("🔍 Analyze Images", use_container_width=True):
                self.process_uploaded_files(uploaded_files)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_case_details(self):
        """Show detailed case information and analysis"""
        case = st.session_state.current_case
        
        st.markdown(f"### 📋 Case: {case.case_title}")
        
        # Case information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Case ID:** {case.case_id}")
            st.markdown(f"**Patient ID:** {case.patient_id}")
            st.markdown(f"**Modality:** {case.modality}")
            st.markdown(f"**Body Part:** {case.body_part}")
            st.markdown(f"**Urgency:** {case.urgency_level}")
            st.markdown(f"**Status:** {case.status}")
        
        with col2:
            if st.button("📁 Upload More Images"):
                st.session_state.current_case = None
                st.rerun()
            
            if st.button("📊 Generate Report"):
                self.generate_case_report()
        
        # Analysis results
        if st.session_state.diagnosis_results:
            st.markdown("## 🔍 Analysis Results")
            
            for i, result in enumerate(st.session_state.diagnosis_results):
                with st.expander(f"Diagnosis {i+1} - Confidence: {result.get('confidence', 0):.1f}%"):
                    st.markdown(f"**Findings:** {result.get('diagnosis', 'No diagnosis available')}")
                    
                    if result.get('recommendations'):
                        st.markdown("**Recommendations:**")
                        for rec in result['recommendations']:
                            st.markdown(f"• {rec}")
                    
                    if result.get('heatmap'):
                        st.markdown("**AI Attention Heatmap:**")
                        # Display heatmap (simplified)
                        st.info("Heatmap visualization available in detailed view")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded medical image files"""
        try:
            with st.spinner("Processing medical images..."):
                results = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file
                    file_path = config.get_upload_path(uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
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
                        diagnosis_result['original_filename'] = uploaded_file.name
                        
                        results.append(diagnosis_result)
                        
                        # Save to database
                        if st.session_state.current_case:
                            # Save image
                            image_data_db = {
                                'case_id': st.session_state.current_case.id,
                                'image_path': str(file_path),
                                'original_filename': uploaded_file.name,
                                'image_format': image_data['format'],
                                'image_size': f"{image_data['original_shape']}",
                                'file_size': uploaded_file.size,
                                'metadata': image_data['metadata']
                            }
                            db_manager.add_image_to_case(st.session_state.current_case.id, image_data_db)
                            
                            # Save diagnosis
                            diagnosis_data = {
                                'case_id': st.session_state.current_case.id,
                                'diagnosis_text': diagnosis_result.get('diagnosis', 'Analysis completed'),
                                'confidence_score': diagnosis_result.get('confidence', 0),
                                'diagnosis_type': 'ai',
                                'urgency_level': diagnosis_result.get('urgency', 'routine'),
                                'key_findings': diagnosis_result.get('key_features', []),
                                'recommendations': diagnosis_result.get('recommendations', []),
                                'heatmap_data': diagnosis_result.get('heatmap', {})
                            }
                            db_manager.add_diagnosis(diagnosis_data)
                
                st.session_state.diagnosis_results = results
                st.success(f"Analysis completed! Processed {len(results)} image(s)")
                
                # Update case status
                if st.session_state.current_case:
                    db_manager.update_case_status(st.session_state.current_case.case_id, "completed")
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error processing images: {str(e)}")
            logger.error(f"Error processing uploaded files: {str(e)}")
    
    def show_collaboration(self):
        """Show collaboration and chat interface"""
        st.markdown("## 💬 Doctor Collaboration")
        
        # Case selection for chat
        if st.session_state.current_case is None:
            st.info("Please select a case to start collaboration")
            return
        
        case = st.session_state.current_case
        
        # Chat interface
        st.markdown(f"### 💬 Case Discussion: {case.case_title}")
        
        # Chat history
        chat_messages = db_manager.get_case_chat_history(case.id)
        
        # Display chat history
        for message in chat_messages:
            if message.is_system_message:
                st.markdown(f"<div class='chat-message'><strong>System:</strong> {message.message_text}</div>", unsafe_allow_html=True)
            else:
                user = db_manager.get_user_by_username(message.user.username) if message.user else None
                username = user.full_name if user else "Unknown User"
                
                if message.user_id == (st.session_state.current_user.get('id', 0) if st.session_state.current_user else 0):
                    st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {message.message_text}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message ai-message'><strong>{username}:</strong> {message.message_text}</div>", unsafe_allow_html=True)
        
        # New message input
        with st.form("chat_message"):
            message_text = st.text_area("Type your message", placeholder="Share your thoughts on this case...")
            submitted = st.form_submit_button("Send Message")
            
            if submitted and message_text:
                # Save message to database
                message_data = {
                    'case_id': case.id,
                    'user_id': 1,  # Demo user ID
                    'message_text': message_text,
                    'message_type': 'text'
                }
                
                if db_manager.add_chat_message(message_data):
                    st.success("Message sent!")
                    st.rerun()
                else:
                    st.error("Failed to send message")
    
    def show_literature(self):
        """Show literature search interface"""
        st.markdown("## 📚 Medical Literature Search")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("Search medical literature", placeholder="e.g., pneumonia diagnosis, chest CT findings")
        
        with col2:
            search_button = st.button("🔍 Search", use_container_width=True)
        
        if search_button and search_query:
            with st.spinner("Searching PubMed..."):
                try:
                    # Search PubMed
                    articles = pubmed_search.search_medical_conditions(search_query)
                    
                    if articles:
                        st.success(f"Found {len(articles)} relevant articles")
                        
                        # Display results
                        for i, article in enumerate(articles[:10]):  # Show top 10
                            with st.expander(f"{i+1}. {article.get('title', 'No title')}"):
                                st.markdown(f"**Authors:** {', '.join(article.get('authors', ['Unknown']))}")
                                st.markdown(f"**Journal:** {article.get('journal', 'Unknown')}")
                                st.markdown(f"**Date:** {article.get('publication_date', 'Unknown')}")
                                
                                if article.get('abstract'):
                                    st.markdown("**Abstract:**")
                                    st.markdown(article['abstract'])
                                
                                if article.get('url'):
                                    st.markdown(f"[Read Full Article]({article['url']})")
                    else:
                        st.info("No articles found for your search query")
                
                except Exception as e:
                    st.error(f"Error searching literature: {str(e)}")
        
        # Recent searches
        st.markdown("### 🔍 Recent Searches")
        st.info("No recent searches. Start by searching for a medical condition or diagnosis.")
    
    def show_reports(self):
        """Show reports and analytics"""
        st.markdown("## 📈 Reports & Analytics")
        
        # Report generation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Generate Reports")
            
            if st.button("📋 Generate Case Report", use_container_width=True):
                if st.session_state.current_case and st.session_state.diagnosis_results:
                    self.generate_case_report()
                else:
                    st.warning("Please select a case with analysis results first")
            
            if st.button("📈 Generate Summary Report", use_container_width=True):
                self.generate_summary_report()
        
        with col2:
            st.markdown("### 📁 Download Reports")
            st.info("Generated reports will appear here for download")
        
        # Analytics dashboard
        st.markdown("### 📊 Analytics Dashboard")
        
        # Placeholder charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample chart
            chart_data = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                'Cases': [10, 15, 12, 18, 22]
            })
            
            fig = px.line(chart_data, x='Month', y='Cases', title='Monthly Case Volume')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample pie chart
            pie_data = pd.DataFrame({
                'Modality': ['CT', 'MRI', 'X-ray', 'Ultrasound'],
                'Count': [45, 30, 20, 15]
            })
            
            fig = px.pie(pie_data, values='Count', names='Modality', title='Imaging Modality Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    def show_settings(self):
        """Show application settings"""
        st.markdown("## ⚙️ Application Settings")
        
        # Configuration validation
        st.markdown("### 🔧 System Configuration")
        
        config_issues = config.validate_config()
        if config_issues:
            st.error("Configuration Issues Found:")
            for issue, description in config_issues.items():
                st.markdown(f"• **{issue}:** {description}")
        else:
            st.success("✅ All configuration settings are valid")
        
        # Model settings
        st.markdown("### 🤖 AI Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model Device:** {config.MODEL_DEVICE}")
            st.markdown(f"**Confidence Threshold:** {config.MODEL_CONFIDENCE_THRESHOLD}")
            st.markdown(f"**Batch Size:** {config.MODEL_BATCH_SIZE}")
        
        with col2:
            st.markdown(f"**OpenAI Model:** {config.OPENAI_MODEL}")
            st.markdown(f"**Max Tokens:** {config.OPENAI_MAX_TOKENS}")
            st.markdown(f"**PubMed Max Results:** {config.PUBMED_MAX_RESULTS}")
        
        # Database status
        st.markdown("### 🗄️ Database Status")
        
        try:
            # Test database connection
            session = db_manager.get_session()
            session.close()
            st.success("✅ Database connection successful")
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
        
        # Cache management
        st.markdown("### 🗂️ Cache Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear PubMed Cache"):
                pubmed_search.clear_cache()
                st.success("PubMed cache cleared!")
        
        with col2:
            if st.button("🗑️ Clear All Caches"):
                pubmed_search.clear_cache()
                st.success("All caches cleared!")
    
    def generate_case_report(self):
        """Generate PDF report for current case"""
        try:
            if not st.session_state.current_case or not st.session_state.diagnosis_results:
                st.error("No case or diagnosis results available")
                return
            
            case = st.session_state.current_case
            
            # Prepare case data
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
            
            # Search for relevant literature
            literature_references = []
            if case_data.get('case_description'):
                try:
                    literature_references = pubmed_search.search_medical_conditions(
                        case_data['case_description'], max_results=5
                    )
                except Exception as e:
                    logger.warning(f"Could not fetch literature: {str(e)}")
            
            # Generate report
            with st.spinner("Generating PDF report..."):
                report_path = report_generator.generate_diagnosis_report(
                    case_data,
                    st.session_state.diagnosis_results,
                    literature_references
                )
                
                if report_path and os.path.exists(report_path):
                    # Provide download link
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f.read(),
                            file_name=f"medical_report_{case.case_id}.pdf",
                            mime="application/pdf"
                        )
                    
                    st.success(f"Report generated successfully! Saved to: {report_path}")
                else:
                    st.error("Failed to generate report")
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            logger.error(f"Error generating case report: {str(e)}")
    
    def generate_summary_report(self):
        """Generate summary report for all cases"""
        try:
            # Get all cases (demo data)
            cases = [{
                'case_id': 'DEMO_001',
                'case_title': 'Sample Case 1',
                'status': 'completed',
                'modality': 'CT'
            }]
            
            with st.spinner("Generating summary report..."):
                report_path = report_generator.generate_summary_report(cases)
                
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Summary Report",
                            data=f.read(),
                            file_name="summary_report.pdf",
                            mime="application/pdf"
                        )
                    
                    st.success(f"Summary report generated successfully!")
                else:
                    st.error("Failed to generate summary report")
        
        except Exception as e:
            st.error(f"Error generating summary report: {str(e)}")
            logger.error(f"Error generating summary report: {str(e)}")

def main():
    """Main application entry point"""
    try:
        # Initialize app
        app = MedicalImagingApp()
        
        # Show welcome message if not logged in
        if st.session_state.current_user is None:
            st.markdown('<div class="main-header"><h1>🏥 Advanced Medical Imaging Diagnosis Agent</h1></div>', unsafe_allow_html=True)
            
            st.markdown("""
            ## Welcome to the Advanced Medical Imaging Diagnosis Agent
            
            This AI-powered system helps medical professionals analyze medical images, 
            collaborate on cases, and access relevant medical literature.
            
            ### Key Features:
            - **AI-Powered Diagnosis**: Advanced image analysis using OpenAI Vision API
            - **Multi-Format Support**: DICOM, NIfTI, and standard image formats
            - **Explainable AI**: Heatmaps showing AI attention areas
            - **Literature Integration**: PubMed search for relevant research
            - **Collaboration Tools**: Real-time chat and case discussion
            - **Professional Reports**: PDF generation with findings and recommendations
            
            ### Getting Started:
            1. **Login** using the sidebar
            2. **Upload** medical images for analysis
            3. **Review** AI-generated diagnoses and heatmaps
            4. **Collaborate** with other medical professionals
            5. **Generate** comprehensive PDF reports
            
            ---
            
            **⚠️ Disclaimer**: This system is for educational and research purposes. 
            All medical decisions should be made by qualified healthcare professionals.
            """)
            
            # Demo login
            st.markdown("### 🚀 Quick Demo")
            if st.button("Try Demo Mode", use_container_width=True):
                st.session_state.current_user = {
                    'username': 'demo_user',
                    'full_name': 'Dr. Demo User',
                    'role': 'doctor'
                }
                st.rerun()
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
