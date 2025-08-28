# Advanced Medical Imaging Diagnosis Agent

A comprehensive AI-powered medical imaging diagnosis system built with Streamlit and FastAPI, featuring advanced image processing, AI analysis, explainable AI (XAI), medical literature integration, and collaborative tools for healthcare professionals.

## 🏥 Features

### Core Functionality
- **Multi-Format Image Support**: DICOM, NIfTI, JPG, PNG, TIFF, BMP
- **AI-Powered Diagnosis**: OpenAI Vision API integration for medical image analysis
- **Explainable AI (XAI)**: Heatmaps showing AI attention areas
- **Medical Literature**: PubMed integration for evidence-based diagnosis
- **Collaboration Tools**: Real-time chat for doctor collaboration
- **Professional Reports**: PDF generation with findings and recommendations

### Technical Features
- **Web Application**: Streamlit-based user interface
- **RESTful API**: FastAPI backend for integration
- **Database Management**: SQLAlchemy ORM with SQLite/PostgreSQL support
- **Image Processing**: OpenCV, PIL, pydicom, nibabel
- **Security**: JWT authentication and API key validation
- **Monitoring**: Health checks and performance metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- PubMed email (for literature search)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Advance-Medical-Imaging-Diagnosis-Agent
```

2. **Install dependencies**
```bash
# For Streamlit application
pip install -r requirements.txt

# For FastAPI application
pip install -r requirements_api.txt
```

3. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your credentials
OPENAI_API_KEY=your_openai_api_key_here
PUBMED_EMAIL=your_email@domain.com
SECRET_KEY=your_secret_key_here
```

4. **Initialize the application**
```bash
# Create necessary directories
mkdir -p uploads models reports cache logs
```

## 📱 Running the Applications

### Streamlit Web Application

```bash
# Start Streamlit app
streamlit run app.py

# Or use the startup script
python start_streamlit.py
```

The Streamlit app will be available at: `http://localhost:8501`

### FastAPI Backend

```bash
# Start FastAPI server
python start_api.py

# Or directly with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI server will be available at: `http://localhost:8000`

**API Documentation**: 
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for AI analysis | Required |
| `PUBMED_EMAIL` | Email for PubMed API access | Required |
| `SECRET_KEY` | Secret key for security | Auto-generated |
| `DATABASE_URL` | Database connection string | `sqlite:///medical_imaging.db` |
| `MODEL_DEVICE` | AI model device (cpu/cuda) | `cpu` |
| `STREAMLIT_SERVER_PORT` | Streamlit server port | `8501` |

### Configuration File

The `config.py` file contains all application settings and can be customized for different environments.

## 📊 API Endpoints

### Health & Status
- `GET /health` - Health check and system status

### User Management
- `POST /users` - Create new user
- `GET /users/{username}` - Get user information

### Case Management
- `POST /cases` - Create new medical case
- `GET /cases` - List cases with filtering
- `GET /cases/{case_id}` - Get case details

### Image Analysis
- `POST /upload` - Upload and analyze medical images
- `POST /reports/generate` - Generate PDF reports

### Literature Search
- `POST /literature/search` - Search medical literature
- `GET /literature/article/{pmid}` - Get article details

### Collaboration
- `POST /cases/{case_id}/chat` - Send chat message
- `GET /cases/{case_id}/chat` - Get chat history

### Analytics
- `GET /analytics/dashboard` - Dashboard statistics

## 🖼️ Supported Image Formats

### Medical Imaging
- **DICOM (.dcm)**: Digital Imaging and Communications in Medicine
- **NIfTI (.nii, .nii.gz)**: Neuroimaging Informatics Technology Initiative

### Standard Formats
- **JPEG (.jpg, .jpeg)**: Joint Photographic Experts Group
- **PNG (.png)**: Portable Network Graphics
- **TIFF (.tiff, .tif)**: Tagged Image File Format
- **BMP (.bmp)**: Bitmap

## 🧠 AI Analysis Features

### Diagnosis Capabilities
- Primary diagnosis identification
- Confidence scoring (0-100%)
- Differential diagnoses
- Key imaging features
- Clinical recommendations
- Urgency level assessment

### Explainable AI
- Attention heatmaps
- Feature importance visualization
- Decision explanation
- Confidence visualization

## 📚 Medical Literature Integration

### PubMed Search
- Medical condition search
- Diagnosis-specific literature
- Recent articles (within time period)
- Systematic reviews and meta-analyses
- Clinical practice guidelines

### Literature Features
- Abstract extraction
- Author information
- Journal details
- Publication dates
- Direct links to full articles

## 🗄️ Database Schema

### Core Tables
- **Users**: Healthcare professionals and administrators
- **Medical Cases**: Patient cases and metadata
- **Medical Images**: Image data and metadata
- **Diagnoses**: AI and human diagnosis results
- **Chat Messages**: Collaboration messages
- **Literature References**: PubMed article references

## 🔒 Security Features

### Authentication
- JWT token-based authentication
- Role-based access control
- API key validation
- Session management

### Data Protection
- Patient data anonymization
- Secure file uploads
- Input validation
- SQL injection prevention

## 📈 Monitoring & Analytics

### Health Monitoring
- System health checks
- Database connectivity
- API response times
- Error tracking

### Analytics Dashboard
- Case volume statistics
- Modality distribution
- Accuracy metrics
- User activity tracking

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Test Structure
- Unit tests for core modules
- Integration tests for API endpoints
- Image processing tests
- Database operation tests

## 🚀 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t medical-imaging-agent .

# Run container
docker run -p 8000:8000 -p 8501:8501 medical-imaging-agent
```

### Production Considerations
- Use production database (PostgreSQL/MySQL)
- Configure proper CORS settings
- Set up SSL/TLS certificates
- Implement rate limiting
- Configure logging and monitoring
- Set up backup and recovery

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Write comprehensive tests

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Medical Disclaimer**: This system is for educational and research purposes only. All medical decisions should be made by qualified healthcare professionals. The AI-generated diagnoses and recommendations should be reviewed and validated by medical experts.

**Security Notice**: This is a demonstration system. For production use in clinical environments, implement appropriate security measures, HIPAA compliance, and medical device regulations.

## 🆘 Support

### Documentation
- API Documentation: Available at `/docs` when running FastAPI
- Code Documentation: Inline docstrings and type hints
- Configuration Guide: See `config.py` and environment variables

### Issues
- Report bugs via GitHub Issues
- Include system information and error logs
- Provide steps to reproduce issues

### Community
- Join discussions in GitHub Discussions
- Contribute to documentation
- Share use cases and feedback

## 🔮 Future Enhancements

### Planned Features
- Real-time collaboration with WebSockets
- Advanced XAI techniques (Grad-CAM, SHAP)
- Multi-modal analysis (text + images)
- Integration with PACS systems
- Mobile application support
- Advanced analytics and reporting

### Research Areas
- Federated learning for privacy-preserving AI
- Multi-center validation studies
- Clinical workflow integration
- Regulatory compliance frameworks

---

**Built with ❤️ for the medical community**

*This project demonstrates the potential of AI in medical imaging while emphasizing the importance of human expertise and clinical validation.*
