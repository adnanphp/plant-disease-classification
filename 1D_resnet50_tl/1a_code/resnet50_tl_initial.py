"""
Transfer Learning for PlantVillage Dataset - Full Dataset Only
Using ResNet50 pretrained on ImageNet (PyTorch Implementation)

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
from torchvision import transforms, models
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


class TransferLearningPlantVillage:
    """
    Transfer Learning for PlantVillage Dataset - Full Dataset Only
    Trains ResNet50 with ImageNet weights using PyTorch
    """
    
    def __init__(self, data_path, output_path='transfer_learning_results'):
        """
        Initialize transfer learning trainer
        
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
        self.epochs = 15  # Transfer learning converges faster
        self.learning_rate = 0.00001
        self.input_size = (224, 224)
        
        logger.info(f"Initialized Transfer Learning trainer with output path: {output_path}")
        logger.info(f"Using device: {device}")
    
    def load_data(self):
        """Load the preprocessed dataset"""
        logger.info("Loading preprocessed dataset...")
        
        # Load split information
        split_path = self.data_path / 'statistics' / 'split_info.pkl'
        if not split_path.exists():
            split_path = Path(self.data_path) / 'split_info.pkl'
        
        with open(split_path, 'rb') as f:
            self.split_info = pickle.load(f)
        
        # Convert paths to strings
        self.split_info['train']['paths'] = [str(p) for p in self.split_info['train']['paths']]
        self.split_info['validation']['paths'] = [str(p) for p in self.split_info['validation']['paths']]
        self.split_info['test']['paths'] = [str(p) for p in self.split_info['test']['paths']]
        
        # Load label encoder
        label_encoder_path = self.data_path / 'statistics' / 'label_encoder.pkl'
        if label_encoder_path.exists():
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
        Create data loaders for full dataset
        """
        # Define transforms
        # Training transforms with mild augmentation for transfer learning
        train_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
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
        
        # Create data loaders
        num_workers = 2
        
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
    
    def build_transfer_model(self):
        """
        Build transfer learning model with ResNet50 pretrained on ImageNet
        """
        # Load pretrained ResNet50
        base_model = models.resnet50(pretrained=True)
        
        logger.info(f"Loaded ResNet50 pretrained on ImageNet")
        logger.info(f"Base model has {len(list(base_model.parameters()))} parameter groups")
        
        # Freeze base model initially
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Replace the classifier head
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        model = base_model
        
        logger.info(f"Custom classification head added:")
        logger.info(f"  - Dense(512) + BatchNorm + ReLU + Dropout(0.5)")
        logger.info(f"  - Dense(256) + BatchNorm + ReLU + Dropout(0.3)")
        logger.info(f"  - Dense({self.num_classes})")
        
        return model
    
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
    
    def train_model(self, model, phase_name, learning_rate=None):
        """
        Train the model
        
        Args:
            model: PyTorch model
            phase_name: 'initial' or 'fine_tune'
            learning_rate: Learning rate (uses default if None)
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Check if model already exists
        model_path = self.models_path / f'resnet50_transfer_full_{phase_name}_best.pth'
        results_path = self.results_path / f'resnet50_transfer_{phase_name}_results.json'
        history_path = self.training_logs_path / f'resnet50_transfer_full_{phase_name}_history.pkl'
        
        if phase_name == 'initial' and model_path.exists() and history_path.exists():
            logger.info(f"✓ {phase_name.capitalize()} model already exists. Loading results...")
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, history, 0  # Return 0 training time since no training done
        
        # Move model to device
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
        
        logger.info(f"Training ResNet50 Transfer Learning - {phase_name} phase...")
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
                # Save model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, model_path)
                logger.info(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        training_time = time.time() - start_time
        
        # Load best model
        if model_path.exists():
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save training history
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        return model, history, training_time
    
    def fine_tune_model(self, model):
        """
        Fine-tune by unfreezing some layers of the base model
        """
        logger.info("\n" + "="*60)
        logger.info("Starting Fine-Tuning Phase")
        logger.info("="*60)
        
        # Check if fine-tuned model already exists
        model_path = self.models_path / 'resnet50_transfer_full_fine_tune_best.pth'
        if model_path.exists():
            logger.info("✓ Fine-tuned model already exists. Loading...")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Load history
            history_path = self.training_logs_path / 'resnet50_transfer_full_fine_tune_history.pkl'
            if history_path.exists():
                with open(history_path, 'rb') as f:
                    history = pickle.load(f)
                return model, history, 0
        
        # Unfreeze the top layers of the base model
        # For ResNet50, unfreeze last 50 layers
        layers_to_unfreeze = 50
        layer_count = 0
        
        for param in model.parameters():
            param.requires_grad = True
            layer_count += 1
            if layer_count > layers_to_unfreeze:
                break
        
        logger.info(f"Unfroze last {layers_to_unfreeze} layers")
        
        # Count trainable parameters after unfreezing
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after unfreezing: {trainable_params/1e6:.2f}M")
        
        # Continue training with lower learning rate
        fine_tuned_model, fine_tune_history, fine_tune_time = self.train_model(
            model, 'fine_tune', learning_rate=self.learning_rate / 10
        )
        
        return fine_tuned_model, fine_tune_history, fine_tune_time
    
    def evaluate_model(self, model, model_name):
        """Evaluate model on test set"""
        logger.info(f"Evaluating {model_name} on test set...")
        
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
            'model_name': model_name,
            'dataset_type': 'full',
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'num_classes': self.num_classes
        }
        
        # Save results
        results_path = self.results_path / f'resnet50_transfer_{model_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_training_history(self, history, phase_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title(f'ResNet50 Transfer Learning - Accuracy ({phase_name})', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title(f'ResNet50 Transfer Learning - Loss ({phase_name})', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(history['lr'], label='Learning Rate', linewidth=2, color='green')
        axes[1, 0].set_title(f'ResNet50 Transfer Learning - Learning Rate ({phase_name})', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Empty subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_path / f'training_history_resnet50_transfer_{phase_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also plot combined (both phases)
        if phase_name == 'fine_tune' and hasattr(self, 'initial_history'):
            self.plot_combined_history()
    
    def plot_combined_history(self):
        """Plot combined training history (initial + fine-tune)"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Combine histories
        initial_acc = self.initial_history['train_acc']
        initial_val_acc = self.initial_history['val_acc']
        fine_tune_acc = self.fine_tune_history['train_acc']
        fine_tune_val_acc = self.fine_tune_history['val_acc']
        
        # Offset epochs for fine-tuning
        initial_epochs = len(initial_acc)
        fine_tune_epochs = len(fine_tune_acc)
        total_epochs = initial_epochs + fine_tune_epochs
        
        # Accuracy plot
        axes[0].plot(range(1, initial_epochs + 1), initial_acc, label='Train (Initial)', linewidth=2)
        axes[0].plot(range(1, initial_epochs + 1), initial_val_acc, label='Validation (Initial)', linewidth=2)
        axes[0].plot(range(initial_epochs + 1, total_epochs + 1), fine_tune_acc, label='Train (Fine-tune)', linewidth=2)
        axes[0].plot(range(initial_epochs + 1, total_epochs + 1), fine_tune_val_acc, label='Validation (Fine-tune)', linewidth=2)
        axes[0].axvline(x=initial_epochs + 0.5, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        axes[0].set_title('ResNet50 Transfer Learning - Complete Training', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        initial_loss = self.initial_history['train_loss']
        initial_val_loss = self.initial_history['val_loss']
        fine_tune_loss = self.fine_tune_history['train_loss']
        fine_tune_val_loss = self.fine_tune_history['val_loss']
        
        axes[1].plot(range(1, initial_epochs + 1), initial_loss, label='Train (Initial)', linewidth=2)
        axes[1].plot(range(1, initial_epochs + 1), initial_val_loss, label='Validation (Initial)', linewidth=2)
        axes[1].plot(range(initial_epochs + 1, total_epochs + 1), fine_tune_loss, label='Train (Fine-tune)', linewidth=2)
        axes[1].plot(range(initial_epochs + 1, total_epochs + 1), fine_tune_val_loss, label='Validation (Fine-tune)', linewidth=2)
        axes[1].axvline(x=initial_epochs + 0.5, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        axes[1].set_title('ResNet50 Transfer Learning - Complete Training', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'training_history_resnet50_transfer_complete.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, conf_matrix, model_name):
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
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(self.figures_path / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_results_summary(self, initial_results, fine_tune_results, param_counts, total_training_time):
        """Create results summary visualization - FIXED version"""
        logger.info("Creating results summary plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Metrics comparison bar chart
        ax1 = axes[0]
        metrics = ['Initial\nAccuracy', 'Fine-tuned\nAccuracy', 'Improvement']
        initial_acc = initial_results['accuracy']
        fine_tune_acc = fine_tune_results['accuracy']
        improvement = fine_tune_acc - initial_acc
        
        values = [initial_acc, fine_tune_acc, improvement]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy / Improvement')
        ax1.set_title('Transfer Learning Performance Improvement')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Training time and parameters
        ax2 = axes[1]
        
        param_data = ['Trainable\nParams', 'Total\nParams', 'Training\nTime (min)']
        param_values = [
            param_counts['trainable'],
            param_counts['total'],
            total_training_time / 60 if total_training_time > 0 else 0
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
        
        plt.suptitle('ResNet50 Transfer Learning Results - PlantVillage Full Dataset', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'transfer_learning_results_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Results summary saved to {self.figures_path / 'transfer_learning_results_summary.png'}")
    
    def visualize_features(self, model):
        """Visualize features using t-SNE"""
        logger.info("Generating t-SNE feature visualization...")
        
        model.eval()
        model = model.to(device)
        
        # Create feature extractor (remove classification head)
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
        plt.title('t-SNE Visualization - ResNet50 Transfer Learning Features', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'tsne_visualization_resnet50_transfer.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("t-SNE visualization saved")
    
    def generate_report(self, initial_results, fine_tune_results, param_counts, total_training_time):
        """Generate final report"""
        logger.info("Generating final report...")
        
        # Create results summary plot
        self.plot_results_summary(initial_results, fine_tune_results, param_counts, total_training_time)
        
        report = []
        report.append("# Transfer Learning Report - PlantVillage Dataset (PyTorch)")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## Dataset Information")
        report.append(f"- **Number of Classes:** {self.num_classes}")
        report.append(f"- **Training Samples:** {len(self.split_info['train']['labels'])}")
        report.append(f"- **Validation Samples:** {len(self.split_info['validation']['labels'])}")
        report.append(f"- **Test Samples:** {len(self.split_info['test']['labels'])}")
        
        report.append(f"\n## Model Architecture")
        report.append(f"- **Base Model:** ResNet50 (pretrained on ImageNet)")
        report.append(f"- **Custom Head:** Dense(512) + BatchNorm + ReLU + Dropout(0.5) → Dense(256) + BatchNorm + ReLU + Dropout(0.3) → Dense({self.num_classes})")
        report.append(f"- **Total Parameters:** {param_counts['total']:.2f}M")
        report.append(f"- **Trainable Parameters:** {param_counts['trainable']:.2f}M")
        report.append(f"- **Framework:** PyTorch")
        report.append(f"- **Device:** {device}")
        
        report.append(f"\n## Training Configuration")
        report.append(f"- **Batch Size:** {self.batch_size}")
        report.append(f"- **Initial Learning Rate:** {self.learning_rate}")
        report.append(f"- **Fine-tune Learning Rate:** {self.learning_rate/10}")
        report.append(f"- **Max Epochs per Phase:** {self.epochs}")
        report.append(f"- **Early Stopping Patience:** 10")
        report.append(f"- **LR Reduce Patience:** 5")
        report.append(f"- **Input Size:** {self.input_size}")
        report.append(f"- **Optimizer:** Adam")
        report.append(f"- **Loss Function:** Cross Entropy Loss")
        
        report.append(f"\n## Results - Initial Training")
        report.append(f"- **Test Accuracy:** {initial_results['accuracy']:.4f} ({initial_results['accuracy']*100:.2f}%)")
        report.append(f"- **Macro Precision:** {initial_results['classification_report']['macro avg']['precision']:.4f}")
        report.append(f"- **Macro Recall:** {initial_results['classification_report']['macro avg']['recall']:.4f}")
        report.append(f"- **Macro F1-Score:** {initial_results['classification_report']['macro avg']['f1-score']:.4f}")
        
        report.append(f"\n## Results - After Fine-Tuning")
        report.append(f"- **Test Accuracy:** {fine_tune_results['accuracy']:.4f} ({fine_tune_results['accuracy']*100:.2f}%)")
        report.append(f"- **Macro Precision:** {fine_tune_results['classification_report']['macro avg']['precision']:.4f}")
        report.append(f"- **Macro Recall:** {fine_tune_results['classification_report']['macro avg']['recall']:.4f}")
        report.append(f"- **Macro F1-Score:** {fine_tune_results['classification_report']['macro avg']['f1-score']:.4f}")
        
        report.append(f"\n## Improvement with Fine-Tuning")
        acc_improvement = fine_tune_results['accuracy'] - initial_results['accuracy']
        report.append(f"- **Accuracy Improvement:** {acc_improvement:.4f} ({acc_improvement*100:.2f}%)")
        report.append(f"- **Total Training Time:** {total_training_time/60:.2f} minutes")
        
        # Per-class performance (top 10)
        report.append(f"\n## Per-Class Performance (After Fine-Tuning)")
        report.append("\n| Class | Precision | Recall | F1-Score | Support |")
        report.append("|-------|-----------|--------|----------|---------|")
        
        # Get performance for all classes
        class_metrics = []
        for class_name, metrics in fine_tune_results['classification_report'].items():
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
        report.append(f"- **Transfer Learning Effectiveness:** ResNet50 pretrained on ImageNet achieved {fine_tune_results['accuracy']*100:.2f}% accuracy")
        report.append(f"- **Fine-Tuning Benefit:** Fine-tuning improved accuracy by {acc_improvement*100:.2f}%")
        report.append(f"- **Training Efficiency:** Transfer learning converges faster than training from scratch")
        report.append(f"- **PyTorch Implementation:** Successfully implemented transfer learning with ResNet50 using PyTorch")
        
        # Recommendations
        report.append(f"\n## Recommendations")
        report.append(f"\nBased on the analysis:")
        report.append(f"1. **Transfer Learning is highly effective** for plant disease classification")
        report.append(f"2. **Fine-tuning is recommended** to adapt pretrained features to the specific dataset")
        report.append(f"3. **ResNet50 provides excellent accuracy** with {param_counts['total']:.1f}M parameters")
        report.append(f"4. **PyTorch implementation** provides flexibility for further experimentation")
        
        # Save report
        report_path = self.results_path / 'transfer_learning_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("TRANSFER LEARNING TRAINING COMPLETE!")
        print("="*80)
        print(f"Initial Test Accuracy: {initial_results['accuracy']:.4f} ({initial_results['accuracy']*100:.2f}%)")
        print(f"Fine-tuned Test Accuracy: {fine_tune_results['accuracy']:.4f} ({fine_tune_results['accuracy']*100:.2f}%)")
        print(f"Improvement: {acc_improvement:.4f} ({acc_improvement*100:.2f}%)")
        print(f"Total Training Time: {total_training_time/60:.2f} minutes")
        print(f"Device Used: {device}")
        print(f"\nResults saved to: {self.results_path}")
        print(f"Model saved to: {self.models_path}")
        print(f"Figures saved to: {self.figures_path}")
        print("="*80)
    
    def run(self):
        """
        Run complete transfer learning pipeline
        """
        logger.info("="*80)
        logger.info("STARTING TRANSFER LEARNING PIPELINE (PYTORCH)")
        logger.info("="*80)
        
        # Step 1: Create data loaders
        print("\n[Step 1] Creating data loaders...")
        self.create_data_loaders()
        
        # Step 2: Build transfer learning model
        print("\n[Step 2] Building ResNet50 transfer learning model...")
        model = self.build_transfer_model()
        
        # Count parameters
        param_counts = self.count_parameters(model)
        logger.info(f"Model parameters: {param_counts}")
        logger.info(f"  - Trainable: {param_counts['trainable']:.2f}M")
        logger.info(f"  - Total: {param_counts['total']:.2f}M")
        
        # Step 3: Initial training (frozen base model)
        print("\n[Step 3] Initial training phase (base model frozen)...")
        model, initial_history, initial_time = self.train_model(model, 'initial')
        self.initial_history = initial_history
        
        # Plot initial training history
        self.plot_training_history(self.initial_history, 'initial')
        
        # Evaluate initial model
        initial_results = self.evaluate_model(model, 'initial')
        
        # Save confusion matrix
        conf_matrix_initial = np.array(initial_results['confusion_matrix'])
        self.plot_confusion_matrix(conf_matrix_initial, 'resnet50_transfer_initial')
        
        # Step 4: Fine-tuning
        print("\n[Step 4] Fine-tuning phase (unfreezing top layers)...")
        fine_tuned_model, fine_tune_history, fine_tune_time = self.fine_tune_model(model)
        self.fine_tune_history = fine_tune_history
        
        # Plot fine-tuning history
        self.plot_training_history(self.fine_tune_history, 'fine_tune')
        
        # Plot combined history
        if hasattr(self, 'initial_history'):
            self.plot_combined_history()
        
        # Step 5: Evaluate fine-tuned model
        print("\n[Step 5] Evaluating fine-tuned model...")
        fine_tune_results = self.evaluate_model(fine_tuned_model, 'finetuned')
        
        # Save confusion matrix
        conf_matrix_finetuned = np.array(fine_tune_results['confusion_matrix'])
        self.plot_confusion_matrix(conf_matrix_finetuned, 'resnet50_transfer_finetuned')
        
        # Step 6: Feature visualization
        print("\n[Step 6] Generating t-SNE visualization...")
        self.visualize_features(fine_tuned_model)
        
        # Step 7: Generate report
        print("\n[Step 7] Generating final report...")
        total_time = initial_time + fine_tune_time
        self.generate_report(initial_results, fine_tune_results, param_counts, total_time)
        
        # Step 8: Save final model
        torch.save(fine_tuned_model.state_dict(), self.models_path / 'resnet50_transfer_full_final.pth')
        logger.info("Final model saved")
        
        return fine_tuned_model, fine_tune_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "processed_plantvillage"
    OUTPUT_PATH = "transfer_learning_results"
    
    # Check if data exists
    if not Path(DATA_PATH).exists():
        print(f"Error: Data path {DATA_PATH} not found!")
        print("Please run preprocessing first.")
        sys.exit(1)
    
    # Create trainer instance
    trainer = TransferLearningPlantVillage(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Run transfer learning pipeline
    model, results = trainer.run()
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE!")
    print(f"All results saved to: {OUTPUT_PATH}")
    print("="*80)
