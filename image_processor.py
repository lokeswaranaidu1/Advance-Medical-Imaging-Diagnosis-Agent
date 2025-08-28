"""
Advanced Medical Image Processing Module
Handles DICOM, NIfTI, and standard image formats
"""

import cv2
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image
import SimpleITK as sitk
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import logging
from config import config

logger = logging.getLogger(__name__)

class MedicalImageProcessor:
    """Advanced medical image processor for DICOM, NIfTI, and standard formats"""
    
    def __init__(self):
        self.supported_formats = config.SUPPORTED_FORMATS
        self.max_image_size = config.MAX_IMAGE_SIZE
        self.dicom_tags = config.DICOM_TAGS
        
    def load_image(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load medical image and return processed data
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing image data, metadata, and format info
        """
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.dcm':
                return self._load_dicom(file_path)
            elif file_extension in ['.nii', '.gz']:
                return self._load_nifti(file_path)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return self._load_standard_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {str(e)}")
            raise
    
    def _load_dicom(self, file_path: Path) -> Dict[str, Any]:
        """Load and process DICOM file"""
        try:
            # Load DICOM file
            dcm = pydicom.dcmread(str(file_path))
            
            # Extract pixel data
            pixel_array = dcm.pixel_array
            
            # Normalize pixel values
            if hasattr(dcm, 'BitsAllocated') and dcm.BitsAllocated == 16:
                pixel_array = pixel_array.astype(np.float32)
                if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                    pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            # Convert to 8-bit for display
            display_array = self._normalize_for_display(pixel_array)
            
            # Extract metadata
            metadata = self._extract_dicom_metadata(dcm)
            
            return {
                'image_data': pixel_array,
                'display_data': display_array,
                'metadata': metadata,
                'format': 'dicom',
                'original_shape': pixel_array.shape,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {file_path}: {str(e)}")
            raise
    
    def _load_nifti(self, file_path: Path) -> Dict[str, Any]:
        """Load and process NIfTI file"""
        try:
            # Load NIfTI file
            nii_img = nib.load(str(file_path))
            
            # Get image data
            image_data = nii_img.get_fdata()
            
            # Handle 4D data (time series)
            if len(image_data.shape) == 4:
                # Take middle time point for now
                middle_time = image_data.shape[3] // 2
                image_data = image_data[:, :, :, middle_time]
            
            # Normalize for display
            display_data = self._normalize_for_display(image_data)
            
            # Extract metadata
            metadata = {
                'header': dict(nii_img.header),
                'affine': nii_img.affine.tolist(),
                'shape': image_data.shape,
                'data_type': str(image_data.dtype)
            }
            
            return {
                'image_data': image_data,
                'display_data': display_data,
                'metadata': metadata,
                'format': 'nifti',
                'original_shape': image_data.shape,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading NIfTI file {file_path}: {str(e)}")
            raise
    
    def _load_standard_image(self, file_path: Path) -> Dict[str, Any]:
        """Load and process standard image formats"""
        try:
            # Load with PIL first
            pil_image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_data = np.array(pil_image)
            
            # Normalize to 0-1 range
            if image_data.dtype == np.uint8:
                image_data = image_data.astype(np.float32) / 255.0
            
            # Extract metadata
            metadata = {
                'size': pil_image.size,
                'mode': pil_image.mode,
                'format': pil_image.format,
                'info': pil_image.info
            }
            
            return {
                'image_data': image_data,
                'display_data': image_data,
                'metadata': metadata,
                'format': 'standard',
                'original_shape': image_data.shape,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading standard image {file_path}: {str(e)}")
            raise
    
    def _normalize_for_display(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image data for display purposes"""
        try:
            # Handle different data types
            if image_data.dtype == np.uint8:
                return image_data
            
            # Normalize to 0-255 range
            min_val = np.min(image_data)
            max_val = np.max(image_data)
            
            if max_val > min_val:
                normalized = ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image_data, dtype=np.uint8)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing image data: {str(e)}")
            return image_data
    
    def _extract_dicom_metadata(self, dcm: pydicom.Dataset) -> Dict[str, Any]:
        """Extract relevant DICOM metadata"""
        metadata = {}
        
        for tag in self.dicom_tags:
            try:
                if hasattr(dcm, tag):
                    value = getattr(dcm, tag)
                    if hasattr(value, '__str__'):
                        metadata[tag] = str(value)
                    else:
                        metadata[tag] = value
            except Exception as e:
                logger.warning(f"Could not extract DICOM tag {tag}: {str(e)}")
        
        # Add additional useful metadata
        additional_tags = ['ImageType', 'Manufacturer', 'InstitutionName', 'StudyDescription']
        for tag in additional_tags:
            try:
                if hasattr(dcm, tag):
                    metadata[tag] = str(getattr(dcm, tag))
            except Exception:
                pass
        
        return metadata
    
    def preprocess_image(self, image_data: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocess image for AI model input
        
        Args:
            image_data: Input image array
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image array
        """
        try:
            if target_size is None:
                target_size = (224, 224)  # Default size for many models
            
            # Resize image
            if len(image_data.shape) == 3:
                # Color image
                resized = cv2.resize(image_data, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)
            else:
                # Grayscale image
                resized = cv2.resize(image_data, target_size[::-1], interpolation=cv2.INTER_LANCZOS4)
                resized = np.expand_dims(resized, axis=-1)
            
            # Normalize to 0-1 range
            if resized.dtype != np.float32:
                resized = resized.astype(np.float32)
            
            if resized.max() > 1.0:
                resized = resized / 255.0
            
            # Ensure proper shape for model input
            if len(resized.shape) == 3 and resized.shape[-1] == 1:
                # Single channel, expand to 3 channels if needed
                resized = np.repeat(resized, 3, axis=-1)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def apply_window_level(self, image_data: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """
        Apply window/level adjustment for medical images
        
        Args:
            image_data: Input image array
            window_center: Window center value
            window_width: Window width value
            
        Returns:
            Windowed image array
        """
        try:
            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2
            
            windowed = np.clip(image_data, min_val, max_val)
            windowed = (windowed - min_val) / (max_val - min_val)
            
            return windowed
            
        except Exception as e:
            logger.error(f"Error applying window/level: {str(e)}")
            return image_data
    
    def create_thumbnail(self, image_data: np.ndarray, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Create thumbnail for display purposes"""
        try:
            if len(image_data.shape) == 3:
                thumbnail = cv2.resize(image_data, size[::-1], interpolation=cv2.INTER_AREA)
            else:
                thumbnail = cv2.resize(image_data, size[::-1], interpolation=cv2.INTER_AREA)
                thumbnail = np.expand_dims(thumbnail, axis=-1)
            
            return thumbnail
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return image_data
    
    def validate_image(self, image_data: np.ndarray) -> bool:
        """Validate image data for processing"""
        try:
            if image_data is None:
                return False
            
            if not isinstance(image_data, np.ndarray):
                return False
            
            if image_data.size == 0:
                return False
            
            if np.isnan(image_data).any() or np.isinf(image_data).any():
                return False
            
            return True
            
        except Exception:
            return False

# Global instance
image_processor = MedicalImageProcessor()
