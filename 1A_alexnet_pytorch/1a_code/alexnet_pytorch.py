"""
Complete CNN Training Pipeline for PlantVillage Dataset using PyTorch
Reference: Hughes & Salathe, 2015 - arXiv:1511.08060

Trains only AlexNet on the full dataset using PyTorch
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


class AlexNet(nn.Module):
    """
    AlexNet architecture from scratch
    
    Reference: Krizhevsky et al., 2012
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1: Convolutional + MaxPooling
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2: Convolutional + MaxPooling
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3: Convolutional
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Convolutional
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5: Convolutional + MaxPooling
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Calculate the flattened feature size
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize weights using Kaiming initialization
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
        x = torch.flatten(x, 1)
        return x


class AlexNetClassifier(nn.Module):
    """
    AlexNet with classifier head
    """
    def __init__(self, num_classes=1000):
        super(AlexNetClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1: Convolutional + MaxPooling
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2: Convolutional + MaxPooling
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3: Convolutional
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Convolutional
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5: Convolutional + MaxPooling
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            # Layer 6: Fully Connected
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            # Layer 7: Fully Connected
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # Layer 8: Output Layer
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PlantVillageCNN:
    """
    Complete CNN training pipeline for PlantVillage dataset using PyTorch
    """
    
    def __init__(self, data_path, output_path='plantvillage_cnn_results'):
        """
        Initialize the CNN trainer
        
        Args:
            data_path: Path to processed PlantVillage data (from preprocessing)
            output_path: Path to save all results
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
        
        # Load the preprocessed data
        self.load_data()
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 15
        self.learning_rate = 0.001
        self.input_size = (224, 224)
        
        logger.info(f"Initialized CNN trainer with output path: {output_path}")
        logger.info(f"Using device: {device}")
        
    def load_data(self):
        """
        Load the preprocessed and split dataset
        """
        logger.info("Loading preprocessed dataset...")
        
        # Load split information
        split_path = self.data_path / 'statistics' / 'split_info.pkl'
        if not split_path.exists():
            # Try alternative path
            split_path = Path(self.data_path) / 'split_info.pkl'
        
        with open(split_path, 'rb') as f:
            self.split_info = pickle.load(f)
        
        # Load label encoder or create new one
        label_encoder_path = self.data_path / 'statistics' / 'label_encoder.pkl'
        if label_encoder_path.exists():
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            self.label_encoder = LabelEncoder()
            all_labels = (self.split_info['train']['labels'] + 
                         self.split_info['validation']['labels'] + 
                         self.split_info['test']['labels'])
            self.label_encoder.fit(all_labels)
            # Save for future use
            with open(self.data_path / 'statistics' / 'label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        self.num_classes = len(self.label_encoder.classes_)
        
        # Encode labels to integers
        self.train_labels_encoded = self.label_encoder.transform(self.split_info['train']['labels'])
        self.val_labels_encoded = self.label_encoder.transform(self.split_info['validation']['labels'])
        self.test_labels_encoded = self.label_encoder.transform(self.split_info['test']['labels'])
        
        logger.info(f"Loaded dataset with {self.num_classes} classes")
        logger.info(f"Training samples: {len(self.split_info['train']['labels'])}")
        logger.info(f"Validation samples: {len(self.split_info['validation']['labels'])}")
        logger.info(f"Test samples: {len(self.split_info['test']['labels'])}")

    def create_data_loaders(self, use_augmentation=True):
        """
        Create data loaders for training, validation, and testing
        
        Args:
            use_augmentation: Whether to use data augmentation for training
        """
        # Define transforms
        # Training transforms with augmentation
        if use_augmentation:
            train_transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.input_size),
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
        
        # Convert paths to strings
        train_paths_str = [str(path) for path in self.split_info['train']['paths']]
        val_paths_str = [str(path) for path in self.split_info['validation']['paths']]
        test_paths_str = [str(path) for path in self.split_info['test']['paths']]
        
        # Create datasets
        train_dataset = PlantVillageDataset(
            train_paths_str, 
            self.train_labels_encoded,
            transform=train_transform
        )
        
        val_dataset = PlantVillageDataset(
            val_paths_str,
            self.val_labels_encoded,
            transform=val_test_transform
        )
        
        test_dataset = PlantVillageDataset(
            test_paths_str,
            self.test_labels_encoded,
            transform=val_test_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info("Data loaders created successfully")
        logger.info(f"  - Train batches: {len(self.train_loader)}")
        logger.info(f"  - Validation batches: {len(self.val_loader)}")
        logger.info(f"  - Test batches: {len(self.test_loader)}")
    
    def count_parameters(self, model):
        """Count trainable and non-trainable parameters in the model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'trainable': trainable_params / 1e6,  # in millions
            'total': total_params / 1e6
        }
    
    def train_epoch(self, model, loader, criterion, optimizer, device):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc='Training', leave=False)):
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
    
    def train_model(self, model, model_name, dataset_type):
        """
        Train a single model
        
        Args:
            model: PyTorch model to train
            model_name: Name of the architecture
            dataset_type: 'full' (for logging)
        """
        # Move model to device
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Create callbacks
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        logger.info(f"Training {model_name} on {dataset_type} dataset...")
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
                torch.save(model.state_dict(), self.models_path / f'{model_name}_{dataset_type}.pth')
                logger.info(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        training_time = time.time() - start_time
        
        # Load best model
        best_model_path = self.models_path / f'{model_name}_{dataset_type}.pth'
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
        
        # Save training history
        history_path = self.training_logs_path / f'{model_name}_{dataset_type}_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        return model, history, training_time
    
    def evaluate_model(self, model, model_name, dataset_type):
        """Evaluate model on test set"""
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
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Save results
        results = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'num_classes': self.num_classes
        }
        
        results_path = self.results_path / f'{model_name}_{dataset_type}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(conf_matrix, model_name, dataset_type)
        
        return results
    
    def plot_confusion_matrix(self, conf_matrix, model_name, dataset_type):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Use only top 20 classes for visualization if too many
        if conf_matrix.shape[0] > 20:
            # Sum rows and columns to get top classes
            row_sums = conf_matrix.sum(axis=1)
            top_indices = np.argsort(row_sums)[-20:]
            conf_matrix = conf_matrix[top_indices][:, top_indices]
            class_names = [self.label_encoder.classes_[i] for i in top_indices]
        else:
            class_names = self.label_encoder.classes_
        
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {model_name} ({dataset_type})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            self.figures_path / f'confusion_matrix_{model_name}_{dataset_type}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    def plot_training_history(self, history, model_name, dataset_type):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history['train_acc'], label='Train Accuracy')
        axes[0].plot(history['val_acc'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy ({dataset_type})')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history['train_loss'], label='Train Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Loss ({dataset_type})')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            self.figures_path / f'training_history_{model_name}_{dataset_type}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    def visualize_features(self, model, model_name, dataset_type):
        """Visualize features using t-SNE"""
        logger.info(f"Generating t-SNE visualization for {model_name}...")
        
        model.eval()
        model = model.to(device)
        
        # Create feature extractor by removing the last layer
        if hasattr(model, 'classifier'):
            # If model has separate classifier
            feature_extractor = nn.Sequential(*list(model.features.children()))
        else:
            # Try to remove the last layer
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
        
        feature_extractor = feature_extractor.to(device)
        
        # Extract features from test set
        features = []
        labels = []
        
        with torch.no_grad():
            for images, label_batch in self.test_loader:
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
        plt.title(f't-SNE Visualization - {model_name} ({dataset_type})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(
            self.figures_path / f'tsne_{model_name}_{dataset_type}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    def run_alexnet_full(self):
        """
        Train only AlexNet on the full dataset
        """
        logger.info("="*80)
        logger.info("TRAINING ALEXNET ON FULL DATASET")
        logger.info("="*80)
        
        # Build AlexNet model with classifier
        model = AlexNetClassifier(num_classes=self.num_classes)
        
        # Count parameters
        param_counts = self.count_parameters(model)
        logger.info(f"Model parameters: {param_counts}")
        logger.info(f"  - Trainable: {param_counts['trainable']:.2f}M")
        logger.info(f"  - Total: {param_counts['total']:.2f}M")
        
        # Train
        trained_model, history, training_time = self.train_model(
            model, 'alexnet', 'full'
        )
        
        # Plot training history
        self.plot_training_history(history, 'alexnet', 'full')
        
        # Evaluate
        results = self.evaluate_model(trained_model, 'alexnet', 'full')
        results['training_time'] = training_time
        results['parameters'] = param_counts
        
        # Feature visualization
        self.visualize_features(trained_model, 'alexnet', 'full')
        
        # Save model
        torch.save(trained_model.state_dict(), self.models_path / 'alexnet_full_final.pth')
        
        # Save results summary
        summary_df = pd.DataFrame([results])
        summary_df.to_csv(self.results_path / 'alexnet_full_summary.csv', index=False)
        
        # Generate report
        self.generate_report(results)
        
        return results
    
    def generate_report(self, results):
        """
        Generate comprehensive report for AlexNet on full dataset
        """
        logger.info("Generating comprehensive report...")
        
        # Create visualization of results
        self.plot_results(results)
        
        # Generate markdown report
        self.generate_markdown_report(results)
        
        return results

    def plot_results(self, results):
        """Create result visualizations"""

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
        ax1.set_title('AlexNet Performance Metrics')

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
            results['training_time'] / 60
        ]

        bars2 = ax2.bar(param_data, param_values, color=['#f39c12', '#e67e22', '#d35400'])
        ax2.set_ylabel('Value')
        ax2.set_title('Model Specifications')

        # Add value labels - CORRECTED VERSION
        for i, (bar, val) in enumerate(zip(bars2, param_values)):
            if i == 2:  # Training time bar
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val:.1f} min', ha='center', va='bottom', fontsize=10)
            else:  # Parameters bars
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val:.1f}M', ha='center', va='bottom', fontsize=10)

        plt.suptitle('AlexNet Training Results - PlantVillage Full Dataset',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'alexnet_results_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_markdown_report(self, results):
        """Generate a comprehensive markdown report for AlexNet"""
        
        report = []
        report.append("# AlexNet Training Report - PlantVillage Dataset (PyTorch)")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## Dataset Information")
        report.append(f"- **Number of Classes:** {self.num_classes}")
        report.append(f"- **Training Samples:** {len(self.split_info['train']['labels'])}")
        report.append(f"- **Validation Samples:** {len(self.split_info['validation']['labels'])}")
        report.append(f"- **Test Samples:** {len(self.split_info['test']['labels'])}")
        
        report.append(f"\n## Model Architecture")
        report.append(f"- **Architecture:** AlexNet (Krizhevsky et al., 2012)")
        report.append(f"- **Input Size:** {self.input_size}")
        report.append(f"- **Total Parameters:** {results['parameters']['total']:.2f}M")
        report.append(f"- **Trainable Parameters:** {results['parameters']['trainable']:.2f}M")
        report.append(f"- **Framework:** PyTorch")
        report.append(f"- **Device:** {device}")
        
        report.append(f"\n## Training Configuration")
        report.append(f"- **Batch Size:** {self.batch_size}")
        report.append(f"- **Learning Rate:** {self.learning_rate}")
        report.append(f"- **Max Epochs:** {self.epochs}")
        report.append(f"- **Early Stopping Patience:** 10")
        report.append(f"- **Optimizer:** Adam")
        report.append(f"- **Loss Function:** Cross Entropy Loss")
        report.append(f"- **Learning Rate Scheduler:** ReduceLROnPlateau")
        
        # Data augmentation
        report.append(f"\n### Data Augmentation")
        report.append(f"- **Random Horizontal Flip:** Yes")
        report.append(f"- **Random Affine:** Yes (translation ±10%)")
        report.append(f"- **Color Jitter:** Yes (brightness ±20%, contrast ±20%)")
        report.append(f"- **Normalization:** ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
        
        report.append(f"\n## Results Summary")
        report.append(f"\n### Overall Metrics")
        report.append(f"- **Test Accuracy:** {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        report.append(f"- **Macro Avg Precision:** {results['classification_report']['macro avg']['precision']:.4f}")
        report.append(f"- **Macro Avg Recall:** {results['classification_report']['macro avg']['recall']:.4f}")
        report.append(f"- **Macro Avg F1-Score:** {results['classification_report']['macro avg']['f1-score']:.4f}")
        report.append(f"- **Training Time:** {results['training_time']/60:.2f} minutes")
        
        # Per-class performance
        report.append(f"\n### Per-Class Performance")
        report.append("\n| Class | Precision | Recall | F1-Score | Support |")
        report.append("|-------|-----------|--------|----------|---------|")
        
        for class_name, metrics in results['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report.append(
                    f"| {class_name[:30]} | "
                    f"{metrics['precision']:.4f} | "
                    f"{metrics['recall']:.4f} | "
                    f"{metrics['f1-score']:.4f} | "
                    f"{metrics['support']} |"
                )
        
        # Best and worst performing classes
        class_f1_scores = []
        for class_name, metrics in results['classification_report'].items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_f1_scores.append((class_name, metrics['f1-score']))
        
        class_f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        report.append(f"\n### Best Performing Classes (Top 5)")
        for i in range(min(5, len(class_f1_scores))):
            report.append(f"- **{class_f1_scores[i][0]}**: F1-Score = {class_f1_scores[i][1]:.4f}")
        
        report.append(f"\n### Worst Performing Classes (Bottom 5)")
        for i in range(1, min(6, len(class_f1_scores) + 1)):
            report.append(f"- **{class_f1_scores[-i][0]}**: F1-Score = {class_f1_scores[-i][1]:.4f}")
        
        # Observations
        report.append(f"\n## Key Observations")
        
        if results['accuracy'] > 0.9:
            report.append(f"- **Excellent Performance:** AlexNet achieved {results['accuracy']*100:.2f}% accuracy, demonstrating strong feature learning capabilities.")
        elif results['accuracy'] > 0.8:
            report.append(f"- **Good Performance:** AlexNet achieved {results['accuracy']*100:.2f}% accuracy with moderate training time.")
        
        report.append(f"- **Training Efficiency:** The model trained in {results['training_time']/60:.1f} minutes with {results['parameters']['total']:.1f}M parameters.")
        report.append(f"- **PyTorch Implementation:** Successfully implemented AlexNet from scratch using PyTorch framework.")
        
        # Recommendations
        report.append(f"\n## Recommendations")
        report.append(f"\nBased on the analysis:")
        report.append(f"1. **AlexNet is effective** for plant disease classification on this dataset")
        report.append(f"2. **Consider transfer learning** if higher accuracy is required (e.g., using pretrained ResNet or EfficientNet)")
        report.append(f"3. **Model size ({results['parameters']['total']:.1f}M parameters)** is moderate and suitable for deployment on standard hardware")
        report.append(f"4. **PyTorch implementation** provides flexibility for further experimentation and customization")
        
        # Save report
        report_path = self.results_path / 'alexnet_full_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE! SUMMARY")
        print("="*80)
        print(f"\nModel: AlexNet (PyTorch)")
        print(f"Dataset: Full PlantVillage")
        print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Training Time: {results['training_time']/60:.2f} minutes")
        print(f"Total Parameters: {results['parameters']['total']:.2f}M")
        print(f"Device Used: {device}")
        print(f"\nDetailed report saved to: {report_path}")
        print("="*80)
    
    def run_complete_pipeline(self):
        """
        Run the complete training pipeline (only AlexNet on full dataset)
        """
        logger.info("="*80)
        logger.info("STARTING ALEXNET TRAINING ON FULL DATASET (PYTORCH)")
        logger.info("="*80)
        
        # Step 1: Create data loaders
        print("\n[Step 1] Creating data loaders...")
        self.create_data_loaders(use_augmentation=True)
        
        # Step 2: Train AlexNet on full dataset
        print("\n[Step 2] Training AlexNet on full dataset...")
        results = self.run_alexnet_full()
        
        logger.info("="*80)
        logger.info("ALEXNET TRAINING FINISHED SUCCESSFULLY!")
        logger.info("="*80)
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution script for AlexNet training on PlantVillage dataset using PyTorch
    """
    
    # Configuration
    DATA_PATH = "processed_plantvillage"  # Path from preprocessing step
    OUTPUT_PATH = "plantvillage_cnn_results"
    
    # Check if data exists
    if not Path(DATA_PATH).exists():
        print(f"Error: Data path {DATA_PATH} not found!")
        print("Please run preprocessing first or update the path.")
        sys.exit(1)
    
    # Create trainer instance
    trainer = PlantVillageCNN(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline()
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE!")
    print(f"All results saved to: {OUTPUT_PATH}")
    print("="*80)
