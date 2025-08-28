"""
Advanced Model Training and Retraining System for Medical Imaging Diagnosis
Includes data augmentation, hyperparameter tuning, and accuracy improvement techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import optuna
from tqdm import tqdm
import wandb
import pickle
import joblib
from datetime import datetime
import shutil

from config import config
from image_processor import image_processor

logger = logging.getLogger(__name__)

class MedicalImageDataset(Dataset):
    """Custom dataset for medical images with augmentation"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, is_training: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
        # Validate data
        assert len(image_paths) == len(labels), "Image paths and labels must have same length"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load and process image
            image_path = self.image_paths[idx]
            image_data = image_processor.load_image(image_path)
            
            if image_data is None:
                # Return a placeholder image if loading fails
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                logger.warning(f"Failed to load image: {image_path}")
            else:
                image = image_data['image_data']
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {idx}: {str(e)}")
            # Return placeholder
            image = torch.zeros((3, 224, 224))
            label = torch.tensor(0, dtype=torch.long)
            return image, label

class AdvancedDataAugmentation:
    """Advanced data augmentation techniques for medical images"""
    
    def __init__(self):
        self.augmentation_config = {
            'rotation_range': (-15, 15),
            'translation_range': (-0.1, 0.1),
            'scale_range': (0.9, 1.1),
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'noise_factor': 0.05,
            'elastic_deformation': True,
            'elastic_alpha': 1.0,
            'elastic_sigma': 50.0
        }
        
    def get_training_transforms(self, image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
        """Get training transforms with augmentation"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=self.augmentation_config['rotation_range']),
            transforms.ColorJitter(
                brightness=self.augmentation_config['brightness_range'],
                contrast=self.augmentation_config['contrast_range']
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=self.augmentation_config['translation_range'],
                scale=self.augmentation_config['scale_range']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_validation_transforms(self, image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
        """Get validation transforms without augmentation"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def apply_elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic deformation to image"""
        try:
            from scipy.ndimage import map_coordinates
            from scipy.ndimage.filters import gaussian_filter
            
            # Generate random displacement fields
            shape = image.shape[:2]
            dx = np.random.randn(*shape) * self.augmentation_config['elastic_alpha']
            dy = np.random.randn(*shape) * self.augmentation_config['elastic_alpha']
            
            # Smooth displacement fields
            dx = gaussian_filter(dx, self.augmentation_config['elastic_sigma'])
            dy = gaussian_filter(dy, self.augmentation_config['elastic_sigma'])
            
            # Create coordinate grid
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            
            # Apply displacement
            x = x + dx
            y = y + dy
            
            # Map coordinates
            if len(image.shape) == 3:
                deformed = np.zeros_like(image)
                for channel in range(image.shape[2]):
                    deformed[:, :, channel] = map_coordinates(
                        image[:, :, channel], [y, x], order=1, mode='reflect'
                    )
            else:
                deformed = map_coordinates(image, [y, x], order=1, mode='reflect')
            
            return deformed
            
        except ImportError:
            logger.warning("Scipy not available, skipping elastic deformation")
            return image
        except Exception as e:
            logger.error(f"Error applying elastic deformation: {str(e)}")
            return image

class MedicalImageModel(nn.Module):
    """Advanced medical image classification model"""
    
    def __init__(self, num_classes: int, model_type: str = "efficientnet", 
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super(MedicalImageModel, self).__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        
        # Load pretrained backbone
        if model_type == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_type == "resnet":
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_type == "densenet":
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=num_features, num_heads=8, batch_first=True)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention if features are 2D
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # Add sequence dimension
            attended_features, _ = self.attention(features, features, features)
            features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Classification
        output = self.classifier(features)
        return output

class ModelTrainer:
    """Advanced model trainer with multiple training strategies"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training configuration
        self.config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'early_stopping_patience': 10,
            'gradient_clipping': 1.0,
            'mixed_precision': True
        }
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['scheduler_patience'],
            factor=self.config['scheduler_factor'],
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        logger.info(f"Model trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, save_path: str = None) -> Dict[str, Any]:
        """Complete training loop"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"New best model saved: {save_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final evaluation
        final_metrics = self.evaluate_model(val_loader)
        
        return {
            'history': self.history,
            'best_val_loss': best_val_loss,
            'final_metrics': final_metrics,
            'epochs_trained': epoch + 1
        }
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model performance"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': total_loss / len(test_loader),
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, path: str):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
        self.history = checkpoint['history']

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, model_class, train_loader, val_loader, device="cuda"):
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def objective(self, trial):
        """Objective function for optimization"""
        
        # Suggest hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        model_type = trial.suggest_categorical('model_type', ['efficientnet', 'resnet', 'densenet'])
        
        try:
            # Create model
            model = self.model_class(
                num_classes=len(set([label for _, label in self.train_loader.dataset])),
                model_type=model_type,
                dropout_rate=dropout_rate
            )
            
            # Create trainer
            trainer = ModelTrainer(model, self.device)
            trainer.config['learning_rate'] = lr
            trainer.config['weight_decay'] = weight_decay
            
            # Train for a few epochs
            result = trainer.train(self.train_loader, self.val_loader, epochs=10)
            
            # Return validation loss as objective
            return result['best_val_loss']
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return float('inf')
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }

class ModelRetrainer:
    """Model retraining with continuous learning capabilities"""
    
    def __init__(self, base_model_path: str, device: str = "cuda"):
        self.base_model_path = base_model_path
        self.device = device
        self.retraining_history = []
        
    def retrain_with_new_data(self, new_data_loader: DataLoader, 
                             epochs: int = 20, learning_rate: float = 1e-5) -> Dict[str, Any]:
        """Retrain model with new data"""
        
        # Load base model
        model = self._load_base_model()
        trainer = ModelTrainer(model, self.device)
        trainer.config['learning_rate'] = learning_rate
        
        # Retrain
        result = trainer.train(new_data_loader, new_data_loader, epochs=epochs)
        
        # Record retraining
        retraining_record = {
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'learning_rate': learning_rate,
            'final_metrics': result['final_metrics'],
            'data_size': len(new_data_loader.dataset)
        }
        
        self.retraining_history.append(retraining_record)
        
        return result
    
    def _load_base_model(self) -> nn.Module:
        """Load the base model for retraining"""
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        
        # Create model with same architecture
        model = MedicalImageModel(
            num_classes=checkpoint['config'].get('num_classes', 2),
            model_type=checkpoint['config'].get('model_type', 'efficientnet')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_retraining_history(self) -> List[Dict[str, Any]]:
        """Get retraining history"""
        return self.retraining_history
    
    def save_retraining_history(self, path: str):
        """Save retraining history"""
        with open(path, 'w') as f:
            json.dump(self.retraining_history, f, indent=2)

class AccuracyImprovementEngine:
    """Engine for improving model accuracy through various techniques"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        
    def ensemble_predictions(self, models: List[nn.Module], data_loader: DataLoader) -> np.ndarray:
        """Generate ensemble predictions from multiple models"""
        all_predictions = []
        
        for model in models:
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    predictions.extend(pred.cpu().numpy())
            
            all_predictions.append(predictions)
        
        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        return ensemble_pred.astype(int)
    
    def test_time_augmentation(self, data_loader: DataLoader, 
                              augmentation_transforms: List) -> np.ndarray:
        """Apply test time augmentation"""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                
                # Original prediction
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                # Augmented predictions
                aug_predictions = [pred]
                for transform in augmentation_transforms:
                    aug_data = transform(data)
                    aug_output = self.model(aug_data)
                    aug_pred = aug_output.argmax(dim=1)
                    aug_predictions.append(aug_pred)
                
                # Average predictions
                avg_pred = torch.stack(aug_predictions).float().mean(dim=0)
                all_predictions.extend(avg_pred.cpu().numpy())
        
        return np.array(all_predictions)
    
    def confidence_calibration(self, data_loader: DataLoader) -> nn.Module:
        """Calibrate model confidence scores"""
        # Temperature scaling for confidence calibration
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LRScheduler(temperature, lr=0.01)
        
        self.model.eval()
        
        for epoch in range(100):
            optimizer.zero_grad()
            total_loss = 0
            
            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    logits = self.model(data)
                    scaled_logits = logits / temperature
                    
                    loss = nn.CrossEntropyLoss()(scaled_logits, target)
                    total_loss += loss.item()
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        return temperature

# Global instances
data_augmentation = AdvancedDataAugmentation()
model_trainer = None  # Will be initialized when needed
hyperparameter_optimizer = None  # Will be initialized when needed
model_retrainer = None  # Will be initialized when needed
accuracy_improvement_engine = None  # Will be initialized when needed
