#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced Medical Imaging Diagnosis Agent
Tests all major components and integrations
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
try:
    from config import config
    from image_processor import image_processor
    from ai_diagnosis import ai_diagnosis
    from literature_search import pubmed_search
    from database import db_manager
    from report_generator import report_generator
    from xai_engine import xai_engine
    from model_training import MedicalImageDataset, AdvancedDataAugmentation
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and modules are available")

class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        self.config = config
    
    def test_config_loading(self):
        """Test that configuration loads properly"""
        self.assertIsNotNone(self.config)
        self.assertTrue(hasattr(self.config, 'OPENAI_API_KEY'))
        self.assertTrue(hasattr(self.config, 'SUPPORTED_FORMATS'))
    
    def test_path_generation(self):
        """Test path generation methods"""
        upload_path = self.config.get_upload_path("test.dcm")
        self.assertIsInstance(upload_path, Path)
        
        report_path = self.config.get_report_path("test_report.pdf")
        self.assertIsInstance(report_path, Path)
    
    def test_config_validation(self):
        """Test configuration validation"""
        issues = self.config.validate_config()
        # Should return list (empty if no issues)
        self.assertIsInstance(issues, list)

class TestImageProcessor(unittest.TestCase):
    """Test image processing functionality"""
    
    def setUp(self):
        self.processor = image_processor
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_image_loading(self):
        """Test image loading functionality"""
        # Test with numpy array
        result = self.processor.load_image(self.test_image)
        self.assertIsNotNone(result)
        self.assertIn('image_data', result)
        self.assertIn('format', result)
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        processed = self.processor.preprocess_image(self.test_image)
        self.assertIsNotNone(processed)
        self.assertIsInstance(processed, np.ndarray)
    
    def test_format_detection(self):
        """Test image format detection"""
        # Test DICOM format detection
        dicom_data = {'format': 'dicom', 'image_data': self.test_image}
        format_type = self.processor.detect_format(dicom_data)
        self.assertEqual(format_type, 'dicom')
    
    def test_metadata_extraction(self):
        """Test metadata extraction"""
        metadata = self.processor.extract_metadata(self.test_image)
        self.assertIsInstance(metadata, dict)
        self.assertIn('dimensions', metadata)
        self.assertIn('data_type', metadata)

class TestAIDiagnosis(unittest.TestCase):
    """Test AI diagnosis functionality"""
    
    def setUp(self):
        self.ai = ai_diagnosis
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    @patch('openai.OpenAI')
    def test_image_analysis(self, mock_openai):
        """Test AI image analysis"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Normal chest X-ray"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        result = self.ai.analyze_image(self.test_image, {})
        self.assertIsNotNone(result)
        self.assertIn('diagnosis', result)
    
    def test_diagnosis_summary(self):
        """Test diagnosis summarization"""
        results = [
            {'diagnosis': 'Normal', 'confidence': 85},
            {'diagnosis': 'Abnormal', 'confidence': 75}
        ]
        
        summary = self.ai.get_diagnosis_summary(results)
        self.assertIsInstance(summary, dict)
        self.assertIn('total_images', summary)
        self.assertIn('average_confidence', summary)
    
    def test_confidence_scoring(self):
        """Test confidence scoring"""
        confidence = self.ai.calculate_confidence_score(self.test_image)
        self.assertIsInstance(confidence, (int, float))
        self.assertTrue(0 <= confidence <= 100)

class TestLiteratureSearch(unittest.TestCase):
    """Test medical literature search functionality"""
    
    def setUp(self):
        self.search = pubmed_search
    
    @patch('biopython.Entrez')
    def test_medical_condition_search(self, mock_entrez):
        """Test medical condition search"""
        # Mock Entrez response
        mock_entrez.esearch.return_value = {'esearchresult': {'idlist': ['12345']}}
        mock_entrez.efetch.return_value = ['Sample article data']
        
        results = self.search.search_medical_conditions("pneumonia", max_results=5)
        self.assertIsInstance(results, list)
    
    def test_search_filtering(self):
        """Test search result filtering"""
        articles = [
            {'title': 'Test Article 1', 'relevance': 0.8},
            {'title': 'Test Article 2', 'relevance': 0.6}
        ]
        
        filtered = self.search.filter_articles_by_relevance(articles, threshold=0.7)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['title'], 'Test Article 1')

class TestDatabase(unittest.TestCase):
    """Test database functionality"""
    
    def setUp(self):
        self.db = db_manager
        self.test_user_data = {
            'username': 'test_user',
            'email': 'test@example.com',
            'full_name': 'Test User',
            'role': 'doctor'
        }
    
    def test_user_creation(self):
        """Test user creation"""
        user = self.db.create_user(self.test_user_data)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, 'test_user')
    
    def test_case_management(self):
        """Test medical case management"""
        case_data = {
            'case_title': 'Test Case',
            'modality': 'CT',
            'body_part': 'Chest'
        }
        
        case = self.db.create_case(case_data)
        self.assertIsNotNone(case)
        self.assertEqual(case.case_title, 'Test Case')
    
    def test_search_functionality(self):
        """Test case search functionality"""
        cases = self.db.search_cases("Test")
        self.assertIsInstance(cases, list)

class TestReportGenerator(unittest.TestCase):
    """Test report generation functionality"""
    
    def setUp(self):
        self.generator = report_generator
        self.test_case_data = {
            'case_id': 'TEST_001',
            'patient_id': 'PAT_001',
            'case_title': 'Test Case'
        }
        self.test_results = [
            {'diagnosis': 'Normal', 'confidence': 85}
        ]
    
    def test_report_generation(self):
        """Test PDF report generation"""
        report_path = self.generator.generate_diagnosis_report(
            self.test_case_data, 
            self.test_results, 
            []
        )
        self.assertIsNotNone(report_path)
        self.assertTrue(Path(report_path).exists())
    
    def test_report_formatting(self):
        """Test report formatting"""
        formatted_text = self.generator.format_diagnosis_text("Test diagnosis")
        self.assertIsInstance(formatted_text, str)
        self.assertIn("Test diagnosis", formatted_text)

class TestXAIEngine(unittest.TestCase):
    """Test Explainable AI engine"""
    
    def setUp(self):
        self.xai = xai_engine
        self.test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        self.test_diagnosis = {'confidence': 85.0}
    
    def test_heatmap_generation(self):
        """Test heatmap generation"""
        heatmap = self.xai.generate_comprehensive_heatmap(
            self.test_image, 
            self.test_diagnosis
        )
        self.assertIsNotNone(heatmap)
        self.assertIn('colored_heatmap', heatmap)
    
    def test_attention_heatmap(self):
        """Test attention-based heatmap"""
        heatmap = self.xai._generate_attention_heatmap(
            self.test_image, 
            self.test_diagnosis
        )
        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap['method'], 'attention_based')
    
    def test_heatmap_comparison(self):
        """Test heatmap comparison"""
        comparison = self.xai.generate_heatmap_comparison(
            self.test_image, 
            self.test_diagnosis
        )
        self.assertIsNotNone(comparison)
        self.assertIn('comparison_heatmaps', comparison)

class TestModelTraining(unittest.TestCase):
    """Test model training components"""
    
    def setUp(self):
        self.augmentation = AdvancedDataAugmentation()
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_data_augmentation(self):
        """Test data augmentation transforms"""
        transforms = self.augmentation.get_training_transforms()
        self.assertIsNotNone(transforms)
        
        # Test transform application
        augmented = transforms(self.test_image)
        self.assertIsNotNone(augmented)
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        image_paths = ['test1.jpg', 'test2.jpg']
        labels = [0, 1]
        
        dataset = MedicalImageDataset(image_paths, labels)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0][1], 0)

class TestIntegration(unittest.TestCase):
    """Test system integration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(self.test_image_path), test_image)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Load image
        image_data = image_processor.load_image(str(self.test_image_path))
        self.assertIsNotNone(image_data)
        
        # 2. Process image
        processed = image_processor.preprocess_image(image_data['image_data'])
        self.assertIsNotNone(processed)
        
        # 3. Generate heatmap
        heatmap = xai_engine.generate_comprehensive_heatmap(
            processed, 
            {'confidence': 80.0}
        )
        self.assertIsNotNone(heatmap)
        
        # 4. Generate report
        case_data = {'case_id': 'TEST_001', 'case_title': 'Test Case'}
        results = [{'diagnosis': 'Test diagnosis', 'confidence': 80.0}]
        
        report_path = report_generator.generate_diagnosis_report(
            case_data, results, []
        )
        self.assertIsNotNone(report_path)
    
    def test_error_handling(self):
        """Test error handling in workflow"""
        # Test with invalid image path
        result = image_processor.load_image("nonexistent.jpg")
        self.assertIsNone(result)
        
        # Test with invalid data
        try:
            xai_engine.generate_comprehensive_heatmap(None, {})
        except Exception as e:
            self.assertIsInstance(e, Exception)

class TestPerformance(unittest.TestCase):
    """Test system performance"""
    
    def test_image_processing_speed(self):
        """Test image processing performance"""
        import time
        
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        start_time = time.time()
        processed = image_processor.preprocess_image(test_image)
        processing_time = time.time() - start_time
        
        self.assertIsNotNone(processed)
        self.assertLess(processing_time, 5.0)  # Should process in under 5 seconds
    
    def test_heatmap_generation_speed(self):
        """Test heatmap generation performance"""
        import time
        
        test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        test_diagnosis = {'confidence': 75.0}
        
        start_time = time.time()
        heatmap = xai_engine.generate_comprehensive_heatmap(
            test_image, 
            test_diagnosis
        )
        generation_time = time.time() - start_time
        
        self.assertIsNotNone(heatmap)
        self.assertLess(generation_time, 10.0)  # Should generate in under 10 seconds

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfiguration,
        TestImageProcessor,
        TestAIDiagnosis,
        TestLiteratureSearch,
        TestDatabase,
        TestReportGenerator,
        TestXAIEngine,
        TestModelTraining,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
