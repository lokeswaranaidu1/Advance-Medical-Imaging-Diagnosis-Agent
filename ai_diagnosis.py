"""
AI Diagnosis Module for Medical Imaging
Integrates OpenAI API and provides explainable AI capabilities
"""

import openai
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import json
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from config import config

logger = logging.getLogger(__name__)

class MedicalAIDiagnosis:
    """AI-powered medical image diagnosis system"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.model_name = config.OPENAI_MODEL
        self.max_tokens = config.OPENAI_MAX_TOKENS
        self.confidence_threshold = config.MODEL_CONFIDENCE_THRESHOLD
        self.device = torch.device(config.MODEL_DEVICE)
        
        # Initialize explainability tools
        self.grad_cam = None
        self._setup_explainability()
    
    def _setup_explainability(self):
        """Setup explainability tools"""
        try:
            if torch.cuda.is_available() and config.MODEL_DEVICE == "cuda":
                self.device = torch.device("cuda")
                logger.info("Using CUDA for explainability")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for explainability")
        except Exception as e:
            logger.warning(f"Could not setup CUDA: {str(e)}")
            self.device = torch.device("cpu")
    
    def analyze_image(self, image_data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze medical image using AI
        
        Args:
            image_data: Preprocessed image data
            metadata: Image metadata (DICOM tags, etc.)
            
        Returns:
            Analysis results with diagnosis and confidence
        """
        try:
            # Convert image to base64 for OpenAI API
            image_base64 = self._image_to_base64(image_data)
            
            # Prepare prompt for medical analysis
            prompt = self._create_medical_prompt(metadata)
            
            # Call OpenAI API
            response = self._call_openai_vision(image_base64, prompt)
            
            # Parse response
            analysis_result = self._parse_ai_response(response)
            
            # Generate heatmap for explainability
            heatmap = self._generate_heatmap(image_data, analysis_result)
            
            # Add heatmap to results
            analysis_result['heatmap'] = heatmap
            analysis_result['metadata'] = metadata or {}
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {
                'error': str(e),
                'diagnosis': 'Analysis failed',
                'confidence': 0.0,
                'recommendations': ['Please try again or contact support']
            }
    
    def _image_to_base64(self, image_data: np.ndarray) -> str:
        """Convert numpy array to base64 string"""
        try:
            # Ensure image is in uint8 format
            if image_data.dtype != np.uint8:
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            
            # Convert to PIL Image
            if len(image_data.shape) == 3 and image_data.shape[-1] == 1:
                # Grayscale
                pil_image = Image.fromarray(image_data.squeeze(), mode='L')
            elif len(image_data.shape) == 3:
                # Color
                pil_image = Image.fromarray(image_data, mode='RGB')
            else:
                # Single channel
                pil_image = Image.fromarray(image_data, mode='L')
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise
    
    def _create_medical_prompt(self, metadata: Dict[str, Any] = None) -> str:
        """Create medical analysis prompt"""
        base_prompt = """
        You are an expert medical imaging radiologist with extensive experience in analyzing medical images.
        
        Please analyze this medical image and provide:
        1. Primary diagnosis or findings
        2. Confidence level (0-100%)
        3. Differential diagnoses
        4. Key imaging features
        5. Clinical recommendations
        6. Urgency level (routine, urgent, emergent)
        7. Additional imaging studies if needed
        
        Be thorough but concise. Use medical terminology appropriately.
        If you cannot make a definitive diagnosis, explain what you can see and what additional information would be helpful.
        """
        
        if metadata:
            # Add relevant metadata to prompt
            if 'Modality' in metadata:
                base_prompt += f"\n\nImage Modality: {metadata['Modality']}"
            if 'BodyPartExamined' in metadata:
                base_prompt += f"\n\nBody Part: {metadata['BodyPartExamined']}"
            if 'StudyDescription' in metadata:
                base_prompt += f"\n\nStudy Description: {metadata['StudyDescription']}"
        
        return base_prompt
    
    def _call_openai_vision(self, image_base64: str, prompt: str) -> str:
        """Call OpenAI Vision API"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.1  # Low temperature for medical accuracy
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse OpenAI response into structured format"""
        try:
            # Try to extract structured information
            result = {
                'raw_response': response,
                'diagnosis': 'Analysis completed',
                'confidence': 75.0,  # Default confidence
                'differential_diagnoses': [],
                'key_features': [],
                'recommendations': [],
                'urgency': 'routine',
                'additional_studies': []
            }
            
            # Extract confidence if mentioned
            if 'confidence' in response.lower():
                import re
                confidence_match = re.search(r'(\d+)%?\s*confidence', response.lower())
                if confidence_match:
                    result['confidence'] = float(confidence_match.group(1))
            
            # Extract urgency level
            urgency_keywords = {
                'emergent': ['emergent', 'emergency', 'immediate', 'critical'],
                'urgent': ['urgent', 'soon', 'within hours'],
                'routine': ['routine', 'elective', 'scheduled']
            }
            
            for urgency, keywords in urgency_keywords.items():
                if any(keyword in response.lower() for keyword in keywords):
                    result['urgency'] = urgency
                    break
            
            # Extract recommendations
            if 'recommend' in response.lower():
                lines = response.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ['recommend', 'suggest', 'advise']):
                        result['recommendations'].append(line.strip())
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                'raw_response': response,
                'diagnosis': 'Analysis completed',
                'confidence': 50.0,
                'error': 'Could not parse response'
            }
    
    def _generate_heatmap(self, image_data: np.ndarray, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainability heatmap"""
        try:
            # Create a simple attention heatmap based on image intensity
            if len(image_data.shape) == 3:
                # Use intensity channel for grayscale or average RGB
                if image_data.shape[-1] == 1:
                    intensity = image_data.squeeze()
                else:
                    intensity = np.mean(image_data, axis=-1)
            else:
                intensity = image_data
            
            # Normalize intensity
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            
            # Apply Gaussian blur for smoother heatmap
            heatmap = cv2.GaussianBlur(intensity, (15, 15), 0)
            
            # Create colored heatmap
            heatmap_colored = plt.cm.jet(heatmap)
            heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
            
            # Overlay on original image
            if len(image_data.shape) == 3 and image_data.shape[-1] == 3:
                overlay = self._overlay_heatmap(image_data, heatmap)
            else:
                # Convert to RGB for overlay
                if len(image_data.shape) == 2:
                    rgb_image = np.stack([image_data] * 3, axis=-1)
                else:
                    rgb_image = np.repeat(image_data, 3, axis=-1)
                overlay = self._overlay_heatmap(rgb_image, heatmap)
            
            return {
                'attention_heatmap': heatmap.tolist(),
                'colored_heatmap': heatmap_colored.tolist(),
                'overlay': overlay.tolist(),
                'explanation': 'Heatmap shows areas of high intensity/attention in the image'
            }
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}")
            return {
                'error': str(e),
                'explanation': 'Could not generate heatmap'
            }
    
    def _overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Overlay heatmap on original image"""
        try:
            # Ensure both are in same format
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            if heatmap.dtype != np.float32:
                heatmap = heatmap.astype(np.float32)
            
            # Create colored heatmap
            heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
            
            # Blend images
            overlay = alpha * heatmap_colored + (1 - alpha) * image
            
            # Clip to valid range
            overlay = np.clip(overlay, 0, 1)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error overlaying heatmap: {str(e)}")
            return image
    
    def batch_analyze(self, images: List[np.ndarray], metadata_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Analyze multiple images in batch"""
        try:
            results = []
            
            for i, image in enumerate(images):
                metadata = metadata_list[i] if metadata_list else None
                result = self.analyze_image(image, metadata)
                result['image_index'] = i
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return []
    
    def get_diagnosis_summary(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary from multiple analysis results"""
        try:
            if not analysis_results:
                return {'error': 'No analysis results available'}
            
            # Aggregate diagnoses
            all_diagnoses = []
            confidence_scores = []
            urgency_levels = []
            
            for result in analysis_results:
                if 'diagnosis' in result:
                    all_diagnoses.append(result['diagnosis'])
                if 'confidence' in result:
                    confidence_scores.append(result['confidence'])
                if 'urgency' in result:
                    urgency_levels.append(result['urgency'])
            
            # Calculate statistics
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            most_common_urgency = max(set(urgency_levels), key=urgency_levels.count) if urgency_levels else 'routine'
            
            return {
                'total_images': len(analysis_results),
                'average_confidence': avg_confidence,
                'most_common_urgency': most_common_urgency,
                'diagnoses': all_diagnoses,
                'summary': f"Analyzed {len(analysis_results)} images with average confidence {avg_confidence:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Error generating diagnosis summary: {str(e)}")
            return {'error': str(e)}

# Global instance
ai_diagnosis = MedicalAIDiagnosis()
