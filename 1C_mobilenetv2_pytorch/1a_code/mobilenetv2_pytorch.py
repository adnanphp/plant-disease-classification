"""
MobileNetV2 Training on Full PlantVillage Dataset using PyTorch
Reference: Hughes & Salathe, 2015 - arXiv:1511.08060
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Scikit-learn imports
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

# For progress tracking
from tqdm import tqdm
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PlantVillageDataset(Dataset):
    """Custom Dataset for PlantVillage images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (encoded as integers)
            transform: Optional transform to be applied
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block for MobileNetV2"""
    
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidualBlock, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expansion_factor)
        
        layers = []
        
        # Expansion phase (1x1 convolution)
        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution (3x3)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection phase (1x1 convolution)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 architecture from scratch
    
    Reference: Sandler et al., 2018 - "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    """
    
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        
        # Define the architecture: (expansion_factor, out_channels, num_blocks, stride)
        inverted_residual_settings = [
            # t, c, n, s
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        
        # Adjust channels based on width multiplier
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        
        # Initial convolution layer
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_settings:
            output_channel = int(c * width_mult)
            
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidualBlock(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        # Final convolution layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        # Convert list to Sequential
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PlantVillageMobileNetV2:
    """
    Train MobileNetV2 on full PlantVillage dataset using PyTorch
    """
    
    def __init__(self, data_path, output_path='mobilenetv2_results'):
        """
        Initialize the trainer
        
        Args:
            data_path: Path to processed PlantVillage data
            output_path: Path to save results
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_path = self.output_path / 'models'
        self.models_path.mkdir(exist_ok=True)
        self.figures_path = self.output_path / 'figures'
        self.figures_path.mkdir(exist_ok=True)
        self.results_path = self.output_path / 'results'
        self.results_path.mkdir(exist_ok=True)
        self.training_logs_path = self.output_path / 'training_logs'
        self.training_logs_path.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 15
        self.learning_rate = 0.001
        self.input_size = (224, 224)
        
        logger.info(f"Initialized MobileNetV2 trainer with output path: {output_path}")
        logger.info(f"Using device: {device}")
        
    def load_data(self):
        """Load the preprocessed dataset"""
        logger.info("Loading preprocessed dataset...")
        
        # Load split information
        split_path = self.data_path / 'statistics' / 'split_info.pkl'
        with open(split_path, 'rb') as f:
            self.split_info = pickle.load(f)
        
        # Convert paths to strings
        self.split_info['train']['paths'] = [str(p) for p in self.split_info['train']['paths']]
        self.split_info['validation']['paths'] = [str(p) for p in self.split_info['validation']['paths']]
        self.split_info['test']['paths'] = [str(p) for p in self.split_info['test']['paths']]
        
        # Load label encoder
        label_encoder_path = self.data_path / 'statistics' / 'label_encoder.pkl'
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.num_classes = len(self.label_encoder.classes_)
        
        # Encode labels to integers
        self.train_labels_encoded = self.label_encoder.transform(self.split_info['train']['labels'])
        self.val_labels_encoded = self.label_encoder.transform(self.split_info['validation']['labels'])
        self.test_labels_encoded = self.label_encoder.transform(self.split_info['test']['labels'])
        
        logger.info(f"Loaded dataset with {self.num_classes} classes")
        logger.info(f"Training samples: {len(self.split_info['train']['labels'])}")
        logger.info(f"Validation samples: {len(self.split_info['validation']['labels'])}")
        logger.info(f"Test samples: {len(self.split_info['test']['labels'])}")
    
    def create_data_loaders(self):
        """
        Create optimized data loaders for full dataset
        """
        # Define transforms
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation and test transforms (no augmentation)
        val_test_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = PlantVillageDataset(
            self.split_info['train']['paths'],
            self.train_labels_encoded,
            transform=train_transform
        )
        
        val_dataset = PlantVillageDataset(
            self.split_info['validation']['paths'],
            self.val_labels_encoded,
            transform=val_test_transform
        )
        
        test_dataset = PlantVillageDataset(
            self.split_info['test']['paths'],
            self.test_labels_encoded,
            transform=val_test_transform
        )
        
        # Create data loaders with optimized settings
        num_workers = 2  # Adjust based on your CPU cores
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Data loaders created successfully")
        logger.info(f"  - Train batches: {len(self.train_loader)}")
        logger.info(f"  - Validation batches: {len(self.val_loader)}")
        logger.info(f"  - Test batches: {len(self.test_loader)}")
        logger.info(f"  - Batch size: {self.batch_size}")
    
    def count_parameters(self, model):
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'trainable': trainable_params / 1e6,
            'total': total_params / 1e6
        }
    
    def train_epoch(self, model, loader, criterion, optimizer, device):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(loader, desc='Training', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, loader, criterion, device):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Validation', leave=False):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self):
        """Train MobileNetV2 on full dataset"""
        logger.info("="*60)
        logger.info("Training MobileNetV2 on full PlantVillage dataset (PyTorch)")
        logger.info("="*60)
        
        # Check if model already exists
        model_path = self.models_path / 'mobilenetv2_full.pth'
        results_path = self.results_path / 'mobilenetv2_full_results.json'
        history_path = self.training_logs_path / 'mobilenetv2_full_history.pkl'
        
        if model_path.exists() and results_path.exists() and history_path.exists():
            logger.info("✓ MobileNetV2 model already exists. Loading results...")
            with open(results_path, 'r') as f:
                results = json.load(f)
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            logger.info(f"  - Test accuracy: {results.get('accuracy', 'N/A')}")
            logger.info(f"  - Best validation accuracy: {max(history.get('val_acc', [0])):.2f}%")
            return results, history
        
        # Create data loaders
        self.create_data_loaders()
        
        # Build model
        logger.info("Building MobileNetV2 architecture...")
        model = MobileNetV2(num_classes=self.num_classes)
        
        # Count parameters
        param_counts = self.count_parameters(model)
        logger.info(f"Model parameters: {param_counts}")
        logger.info(f"  - Trainable: {param_counts['trainable']:.2f}M")
        logger.info(f"  - Total: {param_counts['total']:.2f}M")
        
        # Move model to device
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        # FIXED: Removed verbose parameter
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, self.train_loader, criterion, optimizer, device)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, self.val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                logger.info(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
        
        # Load best model
        if model_path.exists():
            model.load_state_dict(torch.load(model_path))
        
        # Save training history
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        # Evaluate
        results = self.evaluate_model(model, training_time, param_counts)
        
        # Save final model
        torch.save(model.state_dict(), self.models_path / 'mobilenetv2_full_final.pth')
        
        # Plot training history
        self.plot_training_history(history)
        
        # Visualize features
        self.visualize_features(model)
        
        logger.info("✓ MobileNetV2 training complete!")
        
        return results, history
    
    def evaluate_model(self, model, training_time, param_counts):
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        model.eval()
        model = model.to(device)
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Precision (macro): {class_report['macro avg']['precision']:.4f}")
        logger.info(f"Recall (macro): {class_report['macro avg']['recall']:.4f}")
        logger.info(f"F1-Score (macro): {class_report['macro avg']['f1-score']:.4f}")
        
        results = {
            'model_name': 'mobilenetv2',
            'dataset_type': 'full',
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'num_classes': self.num_classes,
            'training_time': training_time,
            'parameters': param_counts
        }
        
        # Save results
        results_path = self.results_path / 'mobilenetv2_full_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(conf_matrix)
        
        return results
    
    def plot_confusion_matrix(self, conf_matrix):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(14, 12))
        
        # Show top 20 classes if too many
        if conf_matrix.shape[0] > 20:
            row_sums = conf_matrix.sum(axis=1)
            top_indices = np.argsort(row_sums)[-20:]
            conf_matrix = conf_matrix[top_indices][:, top_indices]
            class_names = [self.label_encoder.classes_[i][:20] + '...' 
                          if len(self.label_encoder.classes_[i]) > 20 
                          else self.label_encoder.classes_[i] 
                          for i in top_indices]
        else:
            class_names = [c[:20] + '...' if len(c) > 20 else c 
                          for c in self.label_encoder.classes_]
        
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={'size': 8}
        )
        plt.title('MobileNetV2 - Confusion Matrix (Full Dataset)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(self.figures_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('MobileNetV2 - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('MobileNetV2 - Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save best metrics
        best_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_acc) + 1
        logger.info(f"Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    
    def visualize_features(self, model):
        """Visualize features using t-SNE"""
        logger.info("Generating t-SNE feature visualization...")
        
        model.eval()
        model = model.to(device)
        
        # Create feature extractor (remove classification layer)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = feature_extractor.to(device)
        
        # Extract features from test set
        features = []
        labels = []
        
        with torch.no_grad():
            for images, label_batch in tqdm(self.test_loader, desc='Extracting features', leave=False):
                images = images.to(device)
                feat = feature_extractor(images)
                feat = feat.view(feat.size(0), -1)  # Flatten
                features.extend(feat.cpu().numpy())
                labels.extend(label_batch.numpy())
                
                if len(features) >= 1000:  # Limit to 1000 samples for t-SNE
                    break
        
        features = np.array(features[:1000])
        labels = np.array(labels[:1000])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='tab20', alpha=0.7, s=50
        )
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization - MobileNetV2 Features', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("t-SNE visualization saved")
    
    def plot_results_summary(self, results, history):
        """Create results summary visualization - FIXED version"""
        logger.info("Creating results summary plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Metrics bar chart
        ax1 = axes[0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            results['accuracy'],
            results['classification_report']['macro avg']['precision'],
            results['classification_report']['macro avg']['recall'],
            results['classification_report']['macro avg']['f1-score']
        ]
        
        bars = ax1.bar(metrics, values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score')
        ax1.set_title('MobileNetV2 Performance Metrics')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Training time and parameters
        ax2 = axes[1]
        
        param_data = ['Trainable\nParams', 'Total\nParams', 'Training\nTime (min)']
        param_values = [
            results['parameters']['trainable'],
            results['parameters']['total'],
            results['training_time'] / 60 if results['training_time'] > 0 else 0
        ]
        
        bars2 = ax2.bar(param_data, param_values, color=['#f39c12', '#e67e22', '#d35400'])
        ax2.set_ylabel('Value')
        ax2.set_title('Model Specifications')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # FIXED: Add value labels using index-based approach
        for i, (bar, val) in enumerate(zip(bars2, param_values)):
            if i == 2:  # Training time bar
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{val:.1f} min', ha='center', va='bottom', fontsize=10)
            else:  # Parameters bars
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{val:.1f}M', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('MobileNetV2 Training Results - PlantVillage Full Dataset', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'mobilenetv2_results_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Results summary saved to {self.figures_path / 'mobilenetv2_results_summary.png'}")
    
    def generate_report(self, results, history):
        """Generate final report"""
        logger.info("Generating final report...")
        
        # Create results summary plot
        self.plot_results_summary(results, history)
        
        report = []
        report.append("# MobileNetV2 Training Report - PlantVillage Dataset (PyTorch)")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## Dataset Information")
        report.append(f"- **Number of Classes:** {self.num_classes}")
        report.append(f"- **Training Samples:** {len(self.split_info['train']['labels'])}")
        report.append(f"- **Validation Samples:** {len(self.split_info['validation']['labels'])}")
        report.append(f"- **Test Samples:** {len(self.split_info['test']['labels'])}")
        
        report.append(f"\n## Model Architecture")
        report.append(f"- **Architecture:** MobileNetV2 (Sandler et al., 2018)")
        report.append(f"- **Input Size:** {self.input_size}")
        report.append(f"- **Total Parameters:** {results['parameters']['total']:.2f}M")
        report.append(f"- **Trainable Parameters:** {results['parameters']['trainable']:.2f}M")
        report.append(f"- **Key Features:** Depthwise separable convolutions, inverted residuals")
        report.append(f"- **Framework:** PyTorch")
        report.append(f"- **Device:** {device}")
        
        report.append(f"\n## Training Configuration")
        report.append(f"- **Batch Size:** {self.batch_size}")
        report.append(f"- **Learning Rate:** {self.learning_rate}")
        report.append(f"- **Max Epochs:** {self.epochs}")
        report.append(f"- **Early Stopping Patience:** 10")
        report.append(f"- **Optimizer:** Adam")
        report.append(f"- **Loss Function:** Cross Entropy Loss")
        report.append(f"- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)")
        
        # Data augmentation
        report.append(f"\n### Data Augmentation")
        report.append(f"- **Random Horizontal Flip:** Yes")
        report.append(f"- **Random Affine:** Yes (translation ±10%)")
        report.append(f"- **Color Jitter:** Yes (brightness ±20%, contrast ±20%)")
        report.append(f"- **Normalization:** ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
        
        report.append(f"\n## Results")
        report.append(f"- **Test Accuracy:** {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        report.append(f"- **Macro Precision:** {results['classification_report']['macro avg']['precision']:.4f}")
        report.append(f"- **Macro Recall:** {results['classification_report']['macro avg']['recall']:.4f}")
        report.append(f"- **Macro F1-Score:** {results['classification_report']['macro avg']['f1-score']:.4f}")
        report.append(f"- **Training Time:** {results['training_time']/60:.2f} minutes")
        
        # Best validation accuracy from history
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        report.append(f"- **Best Validation Accuracy:** {best_val_acc:.2f}% (epoch {best_epoch})")
        
        # Per-class performance (top 10)
        report.append(f"\n## Per-Class Performance (Top 10 by F1-Score)")
        report.append("\n| Class | Precision | Recall | F1-Score | Support |")
        report.append("|-------|-----------|--------|----------|---------|")
        
        # Get performance for all classes
        class_metrics = []
        for class_name, metrics in results['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_metrics.append((class_name, metrics))
        
        # Sort by F1-score and show top 10
        class_metrics.sort(key=lambda x: x[1]['f1-score'], reverse=True)
        
        for class_name, metrics in class_metrics[:10]:
            short_name = class_name[:30] + '...' if len(class_name) > 30 else class_name
            report.append(f"| {short_name} | {metrics['precision']:.4f} | "
                         f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
                         f"{metrics['support']} |")
        
        if len(self.label_encoder.classes_) > 10:
            report.append(f"| ... and {len(self.label_encoder.classes_)-10} more classes | ... | ... | ... | ... |")
        
        # Best and worst performing classes
        report.append(f"\n### Best Performing Classes (Top 5)")
        for i in range(min(5, len(class_metrics))):
            report.append(f"- **{class_metrics[i][0]}**: F1-Score = {class_metrics[i][1]['f1-score']:.4f}")
        
        report.append(f"\n### Worst Performing Classes (Bottom 5)")
        for i in range(1, min(6, len(class_metrics) + 1)):
            report.append(f"- **{class_metrics[-i][0]}**: F1-Score = {class_metrics[-i][1]['f1-score']:.4f}")
        
        # Observations
        report.append(f"\n## Key Observations")
        
        if results['accuracy'] > 0.9:
            report.append(f"- **Excellent Performance:** MobileNetV2 achieved {results['accuracy']*100:.2f}% accuracy with only {results['parameters']['total']:.1f}M parameters.")
        elif results['accuracy'] > 0.8:
            report.append(f"- **Good Performance:** MobileNetV2 achieved {results['accuracy']*100:.2f}% accuracy with moderate training time.")
        
        report.append(f"- **Lightweight Architecture:** With only {results['parameters']['total']:.1f}M parameters, MobileNetV2 is highly efficient for deployment on edge devices.")
        report.append(f"- **Training Efficiency:** The model trained in {results['training_time']/60:.1f} minutes, demonstrating fast convergence.")
        report.append(f"- **PyTorch Implementation:** Successfully implemented MobileNetV2 from scratch using PyTorch framework.")
        
        # Recommendations
        report.append(f"\n## Recommendations")
        report.append(f"\nBased on the analysis:")
        report.append(f"1. **MobileNetV2 is ideal for deployment** due to its small size ({results['parameters']['total']:.1f}M parameters)")
        report.append(f"2. **Excellent accuracy-to-parameter ratio** - Great for mobile and edge applications")
        report.append(f"3. **Consider transfer learning** if higher accuracy is required (though at the cost of larger model size)")
        report.append(f"4. **PyTorch implementation** provides flexibility for further experimentation and optimization")
        
        # Save report
        report_path = self.results_path / 'final_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("MOBILENETV2 TRAINING COMPLETE!")
        print("="*80)
        print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Training Time: {results['training_time']/60:.2f} minutes")
        print(f"Total Parameters: {results['parameters']['total']:.2f}M")
        print(f"Device Used: {device}")
        print(f"\nResults saved to: {self.results_path}")
        print(f"Model saved to: {self.models_path}")
        print(f"Figures saved to: {self.figures_path}")
        print("="*80)
    
    def run_complete_pipeline(self):
        """
        Run the complete training pipeline (only MobileNetV2 on full dataset)
        """
        logger.info("="*80)
        logger.info("STARTING MOBILENETV2 TRAINING ON FULL DATASET (PYTORCH)")
        logger.info("="*80)
        
        # Step 1: Create data loaders
        print("\n[Step 1] Creating data loaders...")
        self.create_data_loaders()
        
        # Step 2: Train MobileNetV2 on full dataset
        print("\n[Step 2] Training MobileNetV2 on full dataset...")
        results, history = self.train_model()
        
        # Step 3: Generate report
        print("\n[Step 3] Generating final report...")
        self.generate_report(results, history)
        
        logger.info("="*80)
        logger.info("MOBILENETV2 TRAINING FINISHED SUCCESSFULLY!")
        logger.info("="*80)
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution script for MobileNetV2 training on PlantVillage dataset using PyTorch
    """
    
    # Configuration
    DATA_PATH = "processed_plantvillage"  # Path from preprocessing step
    OUTPUT_PATH = "mobilenetv2_results"
    
    # Check if data exists
    if not Path(DATA_PATH).exists():
        print(f"Error: Data path {DATA_PATH} not found!")
        print("Please run preprocessing first.")
        sys.exit(1)
    
    # Create trainer instance
    trainer = PlantVillageMobileNetV2(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline()
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE!")
    print(f"All results saved to: {OUTPUT_PATH}")
    print("="*80)
