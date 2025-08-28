"""
PDF Report Generator for Medical Imaging Diagnosis
Creates professional medical reports with findings, images, and recommendations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

class MedicalReportGenerator:
    """Generate professional medical imaging diagnosis reports"""
    
    def __init__(self):
        self.page_size = A4
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for medical reports"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=HexColor('#2E86AB')
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=HexColor('#A23B72')
        )
        
        # Subsection style
        self.subsection_style = ParagraphStyle(
            'CustomSubsection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=HexColor('#F18F01')
        )
        
        # Normal text style
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        # Important text style
        self.important_style = ParagraphStyle(
            'CustomImportant',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT,
            textColor=HexColor('#C73E1D')
        )
    
    def generate_diagnosis_report(self, case_data: Dict[str, Any], 
                                diagnosis_results: List[Dict[str, Any]],
                                literature_references: List[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive diagnosis report
        
        Args:
            case_data: Case information
            diagnosis_results: AI and human diagnosis results
            literature_references: PubMed literature references
            
        Returns:
            Path to generated PDF report
        """
        try:
            # Create report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            case_id = case_data.get('case_id', 'unknown')
            report_filename = f"medical_report_{case_id}_{timestamp}.pdf"
            report_path = config.get_report_path(report_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(str(report_path), pagesize=self.page_size)
            story = []
            
            # Add header
            story.extend(self._create_header(case_data))
            
            # Add case information
            story.extend(self._create_case_section(case_data))
            
            # Add diagnosis results
            story.extend(self._create_diagnosis_section(diagnosis_results))
            
            # Add images and heatmaps
            story.extend(self._create_images_section(diagnosis_results))
            
            # Add literature references
            if literature_references:
                story.extend(self._create_literature_section(literature_references))
            
            # Add recommendations
            story.extend(self._create_recommendations_section(diagnosis_results))
            
            # Add footer
            story.extend(self._create_footer())
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Medical report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating diagnosis report: {str(e)}")
            raise
    
    def _create_header(self, case_data: Dict[str, Any]) -> List:
        """Create report header"""
        elements = []
        
        # Title
        title = Paragraph("Medical Imaging Diagnosis Report", self.title_style)
        elements.append(title)
        
        # Report metadata
        metadata_data = [
            ["Report Date:", datetime.now().strftime("%B %d, %Y")],
            ["Case ID:", case_data.get('case_id', 'N/A')],
            ["Patient ID:", case_data.get('patient_id', 'N/A')],
            ["Modality:", case_data.get('modality', 'N/A')],
            ["Body Part:", case_data.get('body_part', 'N/A')]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        elements.append(metadata_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_case_section(self, case_data: Dict[str, Any]) -> List:
        """Create case information section"""
        elements = []
        
        # Section header
        section_header = Paragraph("Case Information", self.section_style)
        elements.append(section_header)
        
        # Case description
        if case_data.get('case_description'):
            description = Paragraph(f"<b>Description:</b> {case_data['case_description']}", self.normal_style)
            elements.append(description)
            elements.append(Spacer(1, 10))
        
        # Clinical information
        clinical_info = []
        if case_data.get('urgency_level'):
            clinical_info.append(f"<b>Urgency Level:</b> {case_data['urgency_level'].title()}")
        if case_data.get('status'):
            clinical_info.append(f"<b>Status:</b> {case_data['status'].replace('_', ' ').title()}")
        
        if clinical_info:
            for info in clinical_info:
                elements.append(Paragraph(info, self.normal_style))
            elements.append(Spacer(1, 10))
        
        return elements
    
    def _create_diagnosis_section(self, diagnosis_results: List[Dict[str, Any]]) -> List:
        """Create diagnosis results section"""
        elements = []
        
        # Section header
        section_header = Paragraph("Diagnosis Results", self.section_style)
        elements.append(section_header)
        
        for i, diagnosis in enumerate(diagnosis_results, 1):
            # Diagnosis header
            diagnosis_type = diagnosis.get('diagnosis_type', 'unknown').title()
            diagnosis_header = Paragraph(f"Diagnosis {i} ({diagnosis_type})", self.subsection_style)
            elements.append(diagnosis_header)
            
            # Diagnosis text
            diagnosis_text = diagnosis.get('diagnosis_text', 'No diagnosis provided')
            diagnosis_para = Paragraph(f"<b>Findings:</b> {diagnosis_text}", self.normal_style)
            elements.append(diagnosis_para)
            
            # Confidence score
            if diagnosis.get('confidence_score'):
                confidence = diagnosis['confidence_score']
                confidence_color = self._get_confidence_color(confidence)
                confidence_style = ParagraphStyle(
                    'Confidence',
                    parent=self.normal_style,
                    textColor=confidence_color
                )
                confidence_text = f"<b>Confidence:</b> {confidence:.1f}%"
                confidence_para = Paragraph(confidence_text, confidence_style)
                elements.append(confidence_para)
            
            # Key findings
            if diagnosis.get('key_findings'):
                findings = diagnosis['key_findings']
                if isinstance(findings, list):
                    findings_text = "<b>Key Features:</b><br/>" + "<br/>".join([f"• {finding}" for finding in findings])
                else:
                    findings_text = f"<b>Key Features:</b> {findings}"
                findings_para = Paragraph(findings_text, self.normal_style)
                elements.append(findings_para)
            
            # Recommendations
            if diagnosis.get('recommendations'):
                recs = diagnosis['recommendations']
                if isinstance(recs, list):
                    recs_text = "<b>Recommendations:</b><br/>" + "<br/>".join([f"• {rec}" for rec in recs])
                else:
                    recs_text = f"<b>Recommendations:</b> {recs}"
                recs_para = Paragraph(recs_text, self.normal_style)
                elements.append(recs_para)
            
            elements.append(Spacer(1, 15))
        
        return elements
    
    def _create_images_section(self, diagnosis_results: List[Dict[str, Any]]) -> List:
        """Create images and heatmaps section"""
        elements = []
        
        # Section header
        section_header = Paragraph("Image Analysis & Heatmaps", self.section_style)
        elements.append(section_header)
        
        for i, diagnosis in enumerate(diagnosis_results):
            if diagnosis.get('heatmap'):
                heatmap = diagnosis['heatmap']
                
                # Create heatmap visualization
                try:
                    heatmap_img = self._create_heatmap_image(heatmap)
                    if heatmap_img:
                        # Add heatmap image
                        img_para = Paragraph(f"<b>Heatmap {i+1}:</b>", self.subsection_style)
                        elements.append(img_para)
                        
                        # Add explanation
                        if heatmap.get('explanation'):
                            explanation = Paragraph(f"<i>{heatmap['explanation']}</i>", self.normal_style)
                            elements.append(explanation)
                        
                        elements.append(Spacer(1, 10))
                        
                except Exception as e:
                    logger.warning(f"Could not create heatmap image: {str(e)}")
                    continue
        
        return elements
    
    def _create_literature_section(self, literature_references: List[Dict[str, Any]]) -> List:
        """Create literature references section"""
        elements = []
        
        # Section header
        section_header = Paragraph("Relevant Literature", self.section_style)
        elements.append(section_header)
        
        for i, ref in enumerate(literature_references[:5], 1):  # Limit to top 5
            # Reference header
            ref_header = Paragraph(f"Reference {i}", self.subsection_style)
            elements.append(ref_header)
            
            # Title
            if ref.get('title'):
                title_text = f"<b>Title:</b> {ref['title']}"
                title_para = Paragraph(title_text, self.normal_style)
                elements.append(title_para)
            
            # Authors
            if ref.get('authors'):
                authors_text = f"<b>Authors:</b> {', '.join(ref['authors'])}"
                authors_para = Paragraph(authors_text, self.normal_style)
                elements.append(authors_para)
            
            # Journal and date
            if ref.get('journal') or ref.get('publication_date'):
                journal_text = f"<b>Journal:</b> {ref.get('journal', 'N/A')} ({ref.get('publication_date', 'N/A')})"
                journal_para = Paragraph(journal_text, self.normal_style)
                elements.append(journal_para)
            
            # Abstract (truncated)
            if ref.get('abstract'):
                abstract = ref['abstract'][:200] + "..." if len(ref['abstract']) > 200 else ref['abstract']
                abstract_text = f"<b>Abstract:</b> {abstract}"
                abstract_para = Paragraph(abstract_text, self.normal_style)
                elements.append(abstract_para)
            
            # URL
            if ref.get('url'):
                url_text = f"<b>Link:</b> <a href='{ref['url']}'>{ref['url']}</a>"
                url_para = Paragraph(url_text, self.normal_style)
                elements.append(url_para)
            
            elements.append(Spacer(1, 10))
        
        return elements
    
    def _create_recommendations_section(self, diagnosis_results: List[Dict[str, Any]]) -> List:
        """Create clinical recommendations section"""
        elements = []
        
        # Section header
        section_header = Paragraph("Clinical Recommendations", self.section_style)
        elements.append(section_header)
        
        # Aggregate recommendations
        all_recommendations = []
        urgency_levels = []
        
        for diagnosis in diagnosis_results:
            if diagnosis.get('recommendations'):
                recs = diagnosis['recommendations']
                if isinstance(recs, list):
                    all_recommendations.extend(recs)
                else:
                    all_recommendations.append(recs)
            
            if diagnosis.get('urgency_level'):
                urgency_levels.append(diagnosis['urgency_level'])
        
        # Overall urgency assessment
        if urgency_levels:
            most_urgent = max(set(urgency_levels), key=urgency_levels.count)
            urgency_text = f"<b>Overall Urgency Assessment:</b> {most_urgent.title()}"
            urgency_para = Paragraph(urgency_text, self.important_style)
            elements.append(urgency_para)
            elements.append(Spacer(1, 10))
        
        # Consolidated recommendations
        if all_recommendations:
            # Remove duplicates while preserving order
            unique_recs = []
            for rec in all_recommendations:
                if rec not in unique_recs:
                    unique_recs.append(rec)
            
            recommendations_text = "<b>Key Recommendations:</b><br/>" + "<br/>".join([f"• {rec}" for rec in unique_recs])
            recommendations_para = Paragraph(recommendations_text, self.normal_style)
            elements.append(recommendations_para)
        else:
            no_recs_text = "No specific recommendations provided. Please consult with a radiologist for detailed clinical guidance."
            no_recs_para = Paragraph(no_recs_text, self.normal_style)
            elements.append(no_recs_para)
        
        return elements
    
    def _create_footer(self) -> List:
        """Create report footer"""
        elements = []
        
        elements.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer_text = """
        <b>Disclaimer:</b> This report is generated by an AI-assisted medical imaging diagnosis system. 
        The findings and recommendations should be reviewed and validated by qualified medical professionals. 
        This report is not a substitute for professional medical judgment and should be used in conjunction 
        with clinical evaluation and other diagnostic information.
        """
        disclaimer_para = Paragraph(disclaimer_text, self.normal_style)
        elements.append(disclaimer_para)
        
        # Generation info
        generation_text = f"<i>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>"
        generation_para = Paragraph(generation_text, self.normal_style)
        elements.append(generation_para)
        
        return elements
    
    def _create_heatmap_image(self, heatmap_data: Dict[str, Any]) -> Optional[BytesIO]:
        """Create heatmap image for PDF inclusion"""
        try:
            if 'overlay' in heatmap_data:
                # Convert overlay data to image
                overlay_array = np.array(heatmap_data['overlay'])
                
                # Normalize to 0-255 range
                if overlay_array.max() <= 1.0:
                    overlay_array = (overlay_array * 255).astype(np.uint8)
                
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(overlay_array)
                ax.set_title('AI Attention Heatmap')
                ax.axis('off')
                
                # Save to BytesIO
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='PNG', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                plt.close()
                
                return img_buffer
                
        except Exception as e:
            logger.warning(f"Could not create heatmap image: {str(e)}")
            return None
    
    def _get_confidence_color(self, confidence: float) -> colors.Color:
        """Get color based on confidence score"""
        if confidence >= 80:
            return colors.green
        elif confidence >= 60:
            return colors.orange
        else:
            return colors.red
    
    def generate_summary_report(self, cases: List[Dict[str, Any]]) -> str:
        """Generate summary report for multiple cases"""
        try:
            # Create summary report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"summary_report_{timestamp}.pdf"
            summary_path = config.get_report_path(summary_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(str(summary_path), pagesize=self.page_size)
            story = []
            
            # Title
            title = Paragraph("Medical Imaging Cases Summary Report", self.title_style)
            story.append(title)
            
            # Summary statistics
            total_cases = len(cases)
            completed_cases = len([c for c in cases if c.get('status') == 'completed'])
            pending_cases = len([c for c in cases if c.get('status') == 'pending'])
            
            stats_data = [
                ["Total Cases:", str(total_cases)],
                ["Completed:", str(completed_cases)],
                ["Pending:", str(pending_cases)],
                ["Completion Rate:", f"{(completed_cases/total_cases*100):.1f}%" if total_cases > 0 else "0%"]
            ]
            
            stats_table = Table(stats_data, colWidths=[2*inch, 1*inch])
            stats_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 20))
            
            # Cases list
            cases_header = Paragraph("Case Details", self.section_style)
            story.append(cases_header)
            
            for case in cases:
                case_text = f"<b>{case.get('case_id', 'N/A')}</b> - {case.get('case_title', 'No Title')}"
                case_para = Paragraph(case_text, self.normal_style)
                story.append(case_para)
                
                status_text = f"Status: {case.get('status', 'Unknown')} | Modality: {case.get('modality', 'N/A')}"
                status_para = Paragraph(status_text, self.normal_style)
                story.append(status_para)
                
                story.append(Spacer(1, 8))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Summary report generated: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            raise

# Global instance
report_generator = MedicalReportGenerator()
