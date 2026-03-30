"""
PlantVillage Dataset Preprocessing and Analysis
Reference: Hughes & Salathe, 2015 - arXiv:1511.08060
Complete preprocessing pipeline with stratified sampling to 40,000 images
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# For progress tracking
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PlantVillagePreprocessor:
    """
    Comprehensive preprocessing class for PlantVillage dataset
    """
    
    def __init__(self, data_path, output_path='processed_plantvillage', target_size=40000):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the PlantVillage dataset
            output_path: Path to save processed data and visualizations
            target_size: Target dataset size after stratified sampling (40,000)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.figures_path = self.output_path / 'figures'
        self.figures_path.mkdir(exist_ok=True)
        self.stats_path = self.output_path / 'statistics'
        self.stats_path.mkdir(exist_ok=True)
        self.sampled_path = self.output_path / 'sampled_dataset'
        self.sampled_path.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_counts = {}
        self.sampled_image_paths = []
        self.sampled_labels = []
        
        logger.info(f"Initialized preprocessor with data path: {data_path}")
        logger.info(f"Target dataset size after stratified sampling: {target_size} images")
    
    def explore_dataset_structure(self):
        """
        Explore and visualize the dataset structure
        """
        logger.info("Exploring dataset structure...")
        
        # Get all class directories
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        self.class_names = sorted([d.name for d in class_dirs])
        
        print(f"\n{'='*60}")
        print(f"PLANTVILLAGE DATASET OVERVIEW")
        print(f"{'='*60}")
        print(f"Total number of classes: {len(self.class_names)}")
        print(f"\nClasses found:")
        for i, class_name in enumerate(self.class_names[:10], 1):
            print(f"  {i:2d}. {class_name}")
        if len(self.class_names) > 10:
            print(f"  ... and {len(self.class_names)-10} more classes")
        
        # Count images per class
        for class_dir in tqdm(class_dirs, desc="Counting images per class"):
            class_name = class_dir.name
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')) + \
                    list(class_dir.glob('*.png')) + list(class_dir.glob('*.PNG')) + \
                    list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.JPEG'))
            self.class_counts[class_name] = len(images)
            
            # Store paths for later use
            for img_path in images:
                self.image_paths.append(img_path)
                self.labels.append(class_name)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'image_path': self.image_paths,
            'label': self.labels
        })
        
        # Basic statistics
        total_images = len(self.image_paths)
        print(f"\n{'='*60}")
        print(f"DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total images: {total_images:,}")
        print(f"Average images per class: {total_images/len(self.class_names):.1f}")
        print(f"Min images in a class: {min(self.class_counts.values())}")
        print(f"Max images in a class: {max(self.class_counts.values())}")
        print(f"Target after stratified sampling: {self.target_size:,}")
        
        return self.class_counts
    
    def apply_stratified_sampling(self):
        """
        Apply stratified random sampling to reduce dataset to target_size (40,000)
        while preserving original class distribution
        
        Reference: [5] Johnson & Chen, 2022 - Stratified sampling for dataset reduction
        """
        logger.info(f"Applying stratified random sampling to reduce dataset to {self.target_size} images...")
        
        # Calculate sampling ratios
        total_images = len(self.image_paths)
        sampling_ratio = self.target_size / total_images
        
        print(f"\n{'='*60}")
        print(f"STRATIFIED RANDOM SAMPLING")
        print(f"{'='*60}")
        print(f"Original dataset size: {total_images:,}")
        print(f"Target dataset size: {self.target_size:,}")
        print(f"Overall sampling ratio: {sampling_ratio:.3f}")
        
        # Sample from each class proportionally
        self.sampled_image_paths = []
        self.sampled_labels = []
        samples_per_class = {}
        
        for class_name in tqdm(self.class_names, desc="Sampling per class"):
            # Get all images for this class
            class_indices = [i for i, label in enumerate(self.labels) if label == class_name]
            class_count = len(class_indices)
            
            # Calculate target samples for this class (preserve distribution)
            target_class_samples = max(1, int(class_count * sampling_ratio))
            
            # Ensure we don't sample more than available
            target_class_samples = min(target_class_samples, class_count)
            
            # Random sampling without replacement
            if class_count > target_class_samples:
                sampled_indices = random.sample(class_indices, target_class_samples)
            else:
                sampled_indices = class_indices  # Take all if class is small
            
            # Store sampled data
            for idx in sampled_indices:
                self.sampled_image_paths.append(self.image_paths[idx])
                self.sampled_labels.append(self.labels[idx])
            
            samples_per_class[class_name] = len(sampled_indices)
        
        # Create sampled DataFrame
        self.sampled_df = pd.DataFrame({
            'image_path': self.sampled_image_paths,
            'label': self.sampled_labels
        })
        
        # Save sampled dataset information
        self.sampled_df.to_csv(self.stats_path / 'sampled_dataset.csv', index=False)
        
        # Statistics after sampling
        print(f"\nSampling Results:")
        print(f"  Final sampled size: {len(self.sampled_image_paths):,}")
        print(f"  Classes preserved: {len(self.sampled_df['label'].unique())}")
        print(f"  Min samples per class after sampling: {min(samples_per_class.values())}")
        print(f"  Max samples per class after sampling: {max(samples_per_class.values())}")
        
        # Verify distribution preservation
        self.verify_sampling_distribution(samples_per_class)
        
        return self.sampled_df
    
    def verify_sampling_distribution(self, samples_per_class):
        """
        Verify that class distribution is preserved after sampling
        """
        # Calculate original and sampled distributions
        original_dist = {cls: self.class_counts[cls] for cls in self.class_names}
        sampled_dist = samples_per_class
        
        # Calculate proportions
        total_original = sum(original_dist.values())
        total_sampled = sum(sampled_dist.values())
        
        original_props = {cls: count/total_original for cls, count in original_dist.items()}
        sampled_props = {cls: count/total_sampled for cls, count in sampled_dist.items()}
        
        # Calculate distribution difference
        differences = []
        for cls in self.class_names:
            diff = abs(original_props[cls] - sampled_props[cls])
            differences.append(diff)
        
        max_diff = max(differences)
        mean_diff = np.mean(differences)
        
        print(f"\nDistribution Preservation Verification:")
        print(f"  Maximum proportion difference: {max_diff:.4f}")
        print(f"  Mean proportion difference: {mean_diff:.4f}")
        print(f"  ✅ Class distribution preserved (max diff < 0.01)" if max_diff < 0.01 
              else f"  ⚠️  Class distribution slightly altered (max diff: {max_diff:.4f})")
        
        # Create comparison plot
        self.plot_distribution_comparison(original_props, sampled_props)
    
    def plot_distribution_comparison(self, original_props, sampled_props):
        """
        Plot comparison of original and sampled distributions
        """
        # Get top 15 classes for visualization
        sorted_classes = sorted(original_props.items(), key=lambda x: x[1], reverse=True)[:15]
        class_names_top = [c[0][:20] + '...' if len(c[0]) > 20 else c[0] for c in sorted_classes]
        original_top = [c[1] for c in sorted_classes]
        sampled_top = [sampled_props[c[0]] for c in sorted_classes]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot comparison
        x = np.arange(len(class_names_top))
        width = 0.35
        
        axes[0].bar(x - width/2, original_top, width, label='Original', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, sampled_top, width, label='Sampled (40k)', color='green', alpha=0.7)
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Proportion')
        axes[0].set_title('Class Distribution: Original vs Sampled (Top 15 Classes)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names_top, rotation=45, ha='right')
        axes[0].legend()
        
        # Scatter plot correlation
        all_original = [original_props[cls] for cls in self.class_names]
        all_sampled = [sampled_props[cls] for cls in self.class_names]
        
        axes[1].scatter(all_original, all_sampled, alpha=0.6)
        
        # Add diagonal line
        max_val = max(max(all_original), max(all_sampled))
        axes[1].plot([0, max_val], [0, max_val], 'r--', label='Perfect preservation')
        
        axes[1].set_xlabel('Original Proportion')
        axes[1].set_ylabel('Sampled Proportion')
        axes[1].set_title('Distribution Preservation Correlation')
        axes[1].legend()
        
        # Calculate correlation
        correlation = np.corrcoef(all_original, all_sampled)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                    transform=axes[1].transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Stratified Sampling Verification', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'sampling_verification.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
    
    def visualize_class_distribution(self):
        """
        Create comprehensive visualizations of class distribution for sampled dataset
        """
        logger.info("Creating class distribution visualizations...")
        
        # Use sampled data if available, otherwise use original
        if hasattr(self, 'sampled_df') and len(self.sampled_df) > 0:
            df_to_use = self.sampled_df
            title_prefix = "Sampled Dataset (40k images)"
        else:
            df_to_use = self.df
            title_prefix = "Original Dataset"
        
        # Get class counts
        class_counts = df_to_use['label'].value_counts()
        class_names_sorted = class_counts.index.tolist()
        counts_sorted = class_counts.values.tolist()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Bar plot of class distribution (top 20)
        ax1 = axes[0, 0]
        bars = ax1.bar(range(min(20, len(counts_sorted))), counts_sorted[:20])
        ax1.set_title(f'{title_prefix}: Top 20 Classes by Image Count', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class Index', fontsize=12)
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.set_xticks(range(min(20, len(counts_sorted))))
        ax1.set_xticklabels([f'{i+1}' for i in range(min(20, len(counts_sorted)))], rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 2. Pie chart of top classes
        ax2 = axes[0, 1]
        top_8 = counts_sorted[:8]
        top_8_labels = [name[:20] + '...' if len(name) > 20 else name for name in class_names_sorted[:8]]
        other_count = sum(counts_sorted[8:])
        if other_count > 0:
            top_8.append(other_count)
            top_8_labels.append(f'Other ({len(counts_sorted)-8} classes)')
        
        wedges, texts, autotexts = ax2.pie(top_8, labels=top_8_labels, autopct='%1.1f%%',
                                           startangle=90, explode=[0.05]*len(top_8))
        ax2.set_title(f'{title_prefix}: Class Distribution (Top 8 Classes + Others)', 
                     fontsize=14, fontweight='bold')
        
        # 3. Histogram of class sizes
        ax3 = axes[1, 0]
        ax3.hist(counts_sorted, bins=30, edgecolor='black', alpha=0.7)
        ax3.set_title(f'{title_prefix}: Distribution of Class Sizes', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Images per Class', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.axvline(np.mean(counts_sorted), color='r', linestyle='--', 
                   label=f"Mean: {np.mean(counts_sorted):.1f}")
        ax3.axvline(np.median(counts_sorted), color='g', linestyle='--', 
                   label=f"Median: {np.median(counts_sorted):.1f}")
        ax3.legend()
        
        # 4. Box plot of class sizes
        ax4 = axes[1, 1]
        ax4.boxplot(counts_sorted, vert=False)
        ax4.set_title(f'{title_prefix}: Class Size Distribution (Box Plot)', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Images per Class', fontsize=12)
        ax4.set_yticks([])
        
        plt.suptitle('PlantVillage Dataset: Class Distribution Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional visualization: Cumulative distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative = np.cumsum(sorted(counts_sorted, reverse=True))
        ax.plot(range(1, len(cumulative)+1), cumulative, 'b-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(range(1, len(cumulative)+1), 0, cumulative, alpha=0.3)
        ax.set_title(f'{title_prefix}: Cumulative Image Count Across Classes', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Classes', fontsize=12)
        ax.set_ylabel('Cumulative Number of Images', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for 80% mark
        eighty_percent = 0.8 * cumulative[-1]
        idx_80 = np.where(cumulative >= eighty_percent)[0][0] + 1
        ax.axhline(y=eighty_percent, color='r', linestyle='--', alpha=0.7)
        ax.axvline(x=idx_80, color='r', linestyle='--', alpha=0.7)
        ax.text(idx_80+2, eighty_percent-500, f'80% of images\n({idx_80} classes)', 
               fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'cumulative_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_image_properties(self, sample_size=500):
        """
        Analyze image properties like dimensions, color channels, etc.
        
        Args:
            sample_size: Number of random images to sample for analysis
        """
        logger.info(f"Analyzing image properties on {sample_size} random samples...")
        
        # Use sampled data if available
        if hasattr(self, 'sampled_df') and len(self.sampled_df) > 0:
            image_paths_to_use = self.sampled_image_paths
        else:
            image_paths_to_use = self.image_paths
        
        # Randomly sample images
        sampled_indices = random.sample(range(len(image_paths_to_use)), 
                                       min(sample_size, len(image_paths_to_use)))
        
        # Initialize properties lists
        widths = []
        heights = []
        channels = []
        file_sizes = []
        formats = []
        modes = []
        
        for idx in tqdm(sampled_indices, desc="Analyzing images"):
            img_path = image_paths_to_use[idx]
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    channels.append(len(img.getbands()))
                    modes.append(img.mode)
                
                # File size
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
                
                # Format
                formats.append(img_path.suffix.lower())
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
        
        # Store statistics
        self.image_stats = {
            'widths': widths,
            'heights': heights,
            'channels': channels,
            'file_sizes': file_sizes,
            'formats': formats,
            'modes': modes,
            'aspect_ratios': [w/h for w, h in zip(widths, heights)]
        }
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Width distribution
        axes[0, 0].hist(widths, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Width (pixels)', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].axvline(np.mean(widths), color='r', linestyle='--', 
                          label=f"Mean: {np.mean(widths):.1f}")
        axes[0, 0].legend()
        
        # 2. Height distribution
        axes[0, 1].hist(heights, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Height (pixels)', fontsize=10)
        axes[0, 1].set_ylabel('Frequency', fontsize=10)
        axes[0, 1].axvline(np.mean(heights), color='r', linestyle='--', 
                          label=f"Mean: {np.mean(heights):.1f}")
        axes[0, 1].legend()
        
        # 3. File size distribution
        axes[0, 2].hist(file_sizes, bins=50, edgecolor='black', alpha=0.7, color='salmon')
        axes[0, 2].set_title('File Size Distribution', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('File Size (KB)', fontsize=10)
        axes[0, 2].set_ylabel('Frequency', fontsize=10)
        axes[0, 2].axvline(np.mean(file_sizes), color='r', linestyle='--', 
                          label=f"Mean: {np.mean(file_sizes):.1f} KB")
        axes[0, 2].legend()
        
        # 4. Aspect ratio distribution
        axes[1, 0].hist(self.image_stats['aspect_ratios'], bins=50, 
                       edgecolor='black', alpha=0.7, color='purple')
        axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Aspect Ratio (width/height)', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        
        # 5. Channels distribution
        channel_counts = Counter(channels)
        axes[1, 1].bar(channel_counts.keys(), channel_counts.values(), 
                      color=['gray', 'green', 'blue'])
        axes[1, 1].set_title('Color Channels Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Channels', fontsize=10)
        axes[1, 1].set_ylabel('Count', fontsize=10)
        axes[1, 1].set_xticks(list(channel_counts.keys()))
        
        # 6. Format distribution
        format_counts = Counter(formats)
        axes[1, 2].bar(format_counts.keys(), format_counts.values(), color='orange')
        axes[1, 2].set_title('Image Format Distribution', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Format', fontsize=10)
        axes[1, 2].set_ylabel('Count', fontsize=10)
        
        plt.suptitle('PlantVillage Dataset: Image Properties Analysis', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_path / 'image_properties.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"IMAGE PROPERTIES SUMMARY")
        print(f"{'='*60}")
        print(f"Width range: {min(widths)} - {max(widths)} pixels (mean: {np.mean(widths):.1f})")
        print(f"Height range: {min(heights)} - {max(heights)} pixels (mean: {np.mean(heights):.1f})")
        print(f"File size range: {min(file_sizes):.1f} - {max(file_sizes):.1f} KB (mean: {np.mean(file_sizes):.1f} KB)")
        print(f"Aspect ratio range: {min(self.image_stats['aspect_ratios']):.3f} - {max(self.image_stats['aspect_ratios']):.3f}")
        print(f"Channel modes: {set(modes)}")
        
        return self.image_stats
    
    def sample_images_visualization(self, samples_per_class=3, max_classes=12):
        """
        Display sample images from different classes
        """
        logger.info("Creating sample images visualization...")
        
        # Use sampled data
        if hasattr(self, 'sampled_df') and len(self.sampled_df) > 0:
            df_to_use = self.sampled_df
        else:
            df_to_use = self.df
        
        # Get class list
        class_list = df_to_use['label'].unique().tolist()
        selected_classes = class_list[:min(max_classes, len(class_list))]
        
        n_cols = samples_per_class
        n_rows = len(selected_classes)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, class_name in enumerate(selected_classes):
            # Get images for this class
            class_images = df_to_use[df_to_use['label'] == class_name]['image_path'].tolist()
            
            # Randomly select samples
            selected_images = random.sample(class_images, min(samples_per_class, len(class_images)))
            
            for j, img_path in enumerate(selected_images):
                try:
                    img = Image.open(img_path)
                    
                    # Display original
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    
                    if j == 0:  # Add class name on the leftmost image
                        class_display = class_name[:30] + '...' if len(class_name) > 30 else class_name
                        axes[i, j].set_ylabel(class_display, fontsize=10, rotation=0, 
                                            labelpad=40, ha='right')
                    
                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error\n{str(e)[:20]}', 
                                  ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.suptitle(f'Sample Images from {len(selected_classes)} PlantVillage Classes', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_path / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Create stratified train/validation/test split
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
        """
        logger.info(f"Creating stratified train/val/test split ({train_ratio}/{val_ratio}/{test_ratio})...")
        
        from sklearn.model_selection import train_test_split
        
        # Use sampled data
        if hasattr(self, 'sampled_df') and len(self.sampled_df) > 0:
            df_to_use = self.sampled_df
        else:
            df_to_use = self.df
        
        X = df_to_use['image_path'].values
        y = df_to_use['label'].values
        
        # First split: train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
        )
        
        # Second split: validation and test from temp
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size), stratify=y_temp, random_state=42
        )
        
        # Create split DataFrame
        self.split_df = pd.DataFrame({
            'split': ['train'] * len(X_train) + ['validation'] * len(X_val) + ['test'] * len(X_test),
            'image_path': list(X_train) + list(X_val) + list(X_test),
            'label': list(y_train) + list(y_val) + list(y_test)
        })
        
        # Save split information
        self.split_df.to_csv(self.stats_path / 'dataset_split.csv', index=False)
        
        # Save split indices for easy loading
        split_info = {
            'train': {'paths': list(X_train), 'labels': list(y_train)},
            'validation': {'paths': list(X_val), 'labels': list(y_val)},
            'test': {'paths': list(X_test), 'labels': list(y_test)}
        }
        
        import pickle
        with open(self.stats_path / 'split_info.pkl', 'wb') as f:
            pickle.dump(split_info, f)
        
        # Print split statistics
        print(f"\n{'='*60}")
        print(f"TRAIN/VAL/TEST SPLIT RESULTS")
        print(f"{'='*60}")
        print(f"Training: {len(X_train)} images ({len(X_train)/len(df_to_use)*100:.1f}%)")
        print(f"Validation: {len(X_val)} images ({len(X_val)/len(df_to_use)*100:.1f}%)")
        print(f"Test: {len(X_test)} images ({len(X_test)/len(df_to_use)*100:.1f}%)")
        print(f"Total: {len(df_to_use)} images")
        
        # Verify stratification
        print(f"\nVerifying stratification (sample classes):")
        sample_classes = random.sample(list(df_to_use['label'].unique()), min(5, len(df_to_use['label'].unique())))
        for class_name in sample_classes:
            train_count = len(self.split_df[(self.split_df['split'] == 'train') & 
                                          (self.split_df['label'] == class_name)])
            val_count = len(self.split_df[(self.split_df['split'] == 'validation') & 
                                         (self.split_df['label'] == class_name)])
            test_count = len(self.split_df[(self.split_df['split'] == 'test') & 
                                          (self.split_df['label'] == class_name)])
            total = train_count + val_count + test_count
            print(f"  {class_name[:30]}: Train={train_count} ({train_count/total*100:.1f}%), "
                  f"Val={val_count} ({val_count/total*100:.1f}%), "
                  f"Test={test_count} ({test_count/total*100:.1f}%)")
        
        # Visualize split
        self.visualize_split()
        
        return self.split_df
    
    def visualize_split(self):
        """
        Visualize the train/val/test split
        """
        if not hasattr(self, 'split_df'):
            logger.warning("No split data available. Run create_train_val_test_split first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Split sizes
        split_counts = self.split_df['split'].value_counts()
        axes[0].bar(split_counts.index, split_counts.values, color=['blue', 'orange', 'green'])
        axes[0].set_title('Dataset Split Sizes', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Split')
        axes[0].set_ylabel('Number of Images')
        
        # Add value labels
        for i, (idx, val) in enumerate(split_counts.items()):
            axes[0].text(i, val, f'{val}\n({val/len(self.split_df)*100:.1f}%)', 
                       ha='center', va='bottom')
        
        # Class distribution per split (sample)
        class_sample = random.sample(list(self.split_df['label'].unique()), 
                                    min(10, len(self.split_df['label'].unique())))
        split_data = []
        for class_name in class_sample:
            for split_name in ['train', 'validation', 'test']:
                count = len(self.split_df[(self.split_df['split'] == split_name) & 
                                        (self.split_df['label'] == class_name)])
                split_data.append({'class': class_name[:15], 'split': split_name, 'count': count})
        
        split_df_plot = pd.DataFrame(split_data)
        pivot_data = split_df_plot.pivot(index='class', columns='split', values='count')
        pivot_data.plot(kind='bar', ax=axes[1], color=['blue', 'orange', 'green'])
        axes[1].set_title('Class Distribution Across Splits (Sample)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Number of Images')
        axes[1].legend()
        
        plt.suptitle('Stratified Dataset Split (70/15/15)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'dataset_split.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_statistics_report(self):
        """
        Generate and save comprehensive statistics report
        """
        logger.info("Generating statistics report...")
        
        report = []
        report.append("="*80)
        report.append("PLANTVILLAGE DATASET PREPROCESSING REPORT")
        report.append("="*80)
        report.append(f"\nReference: Hughes & Salathe, 2015 - arXiv:1511.08060")
        report.append(f"\nReport Generated: {pd.Timestamp.now()}")
        
        # Dataset overview
        report.append(f"\n{'='*40}")
        report.append("DATASET OVERVIEW")
        report.append(f"{'='*40}")
        report.append(f"Total Classes (Original): {len(self.class_names)}")
        report.append(f"Total Images (Original): {len(self.image_paths):,}")
        
        if hasattr(self, 'sampled_df') and len(self.sampled_df) > 0:
            report.append(f"\nSTRATIFIED SAMPLING (Target: {self.target_size:,})")
            report.append(f"{'='*40}")
            report.append(f"Final Sampled Size: {len(self.sampled_df):,}")
            report.append(f"Classes Preserved: {len(self.sampled_df['label'].unique())}")
            
            # Original vs sampled distribution statistics
            original_counts = list(self.class_counts.values())
            sampled_counts = self.sampled_df['label'].value_counts().values
            
            report.append(f"\nDistribution Statistics:")
            report.append(f"  Original - Mean: {np.mean(original_counts):.1f}, "
                         f"Std: {np.std(original_counts):.1f}")
            report.append(f"  Sampled  - Mean: {np.mean(sampled_counts):.1f}, "
                         f"Std: {np.std(sampled_counts):.1f}")
        
        # Image properties (if analyzed)
        if hasattr(self, 'image_stats') and self.image_stats:
            report.append(f"\n{'='*40}")
            report.append("IMAGE PROPERTIES (Sample Analysis)")
            report.append(f"{'='*40}")
            
            widths = self.image_stats['widths']
            heights = self.image_stats['heights']
            file_sizes = self.image_stats['file_sizes']
            
            report.append(f"\nDimensions:")
            report.append(f"  Width range: {min(widths)} - {max(widths)} px (mean: {np.mean(widths):.1f})")
            report.append(f"  Height range: {min(heights)} - {max(heights)} px (mean: {np.mean(heights):.1f})")
            
            report.append(f"\nFile Sizes:")
            report.append(f"  Range: {min(file_sizes):.1f} - {max(file_sizes):.1f} KB")
            report.append(f"  Mean: {np.mean(file_sizes):.1f} KB")
            report.append(f"  Std: {np.std(file_sizes):.1f} KB")
            
            # Format distribution
            format_counts = Counter(self.image_stats['formats'])
            report.append(f"\nImage Formats:")
            for fmt, count in format_counts.most_common():
                report.append(f"  {fmt}: {count} ({count/len(self.image_stats['formats'])*100:.1f}%)")
        
        # Split information
        if hasattr(self, 'split_df'):
            report.append(f"\n{'='*40}")
            report.append("DATASET SPLIT (70/15/15)")
            report.append(f"{'='*40}")
            split_counts = self.split_df['split'].value_counts()
            for split_name in ['train', 'validation', 'test']:
                count = split_counts.get(split_name, 0)
                report.append(f"  {split_name.capitalize()}: {count} images "
                             f"({count/len(self.split_df)*100:.1f}%)")
        
        # Recommendations
        report.append(f"\n{'='*40}")
        report.append("PREPROCESSING RECOMMENDATIONS")
        report.append(f"{'='*40}")
        report.append(f"\nBased on the analysis, we recommend:")
        report.append(f"1. Target Size: 224x224 (standard for most CNNs)")
        report.append(f"2. Normalization: [0,1] or [-1,1] range")
        report.append(f"3. Data Augmentation:")
        report.append(f"   - Random horizontal flips")
        report.append(f"   - Random rotations (±15°)")
        report.append(f"   - Random brightness/contrast adjustments")
        report.append(f"4. Use stratified 70/15/15 split as created")
        
        # Save report
        report_path = self.stats_path / 'preprocessing_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Report saved to {report_path}")
        
        # Print report summary
        print('\n'.join(report))
    
    def run_complete_analysis(self):
        """
        Run complete preprocessing analysis pipeline
        """
        logger.info("Starting complete PlantVillage dataset analysis...")
        
        print("\n" + "="*80)
        print("PLANTVILLAGE DATASET PREPROCESSING PIPELINE")
        print("="*80)
        print("Reference: Hughes & Salathe, 2015 - arXiv:1511.08060")
        print("="*80)
        
        # Step 1: Explore dataset structure
        print("\n[Step 1] Exploring dataset structure...")
        self.explore_dataset_structure()
        
        # Step 2: Apply stratified sampling to 40,000 images
        print("\n[Step 2] Applying stratified random sampling to 40,000 images...")
        self.apply_stratified_sampling()
        
        # Step 3: Visualize class distribution (for sampled dataset)
        print("\n[Step 3] Visualizing class distribution...")
        self.visualize_class_distribution()
        
        # Step 4: Analyze image properties
        print("\n[Step 4] Analyzing image properties...")
        self.analyze_image_properties(sample_size=500)
        
        # Step 5: Display sample images
        print("\n[Step 5] Displaying sample images...")
        self.sample_images_visualization(samples_per_class=3, max_classes=8)
        
        # Step 6: Create train/val/test split
        print("\n[Step 6] Creating stratified train/val/test split (70/15/15)...")
        self.create_train_val_test_split()
        
        # Step 7: Generate statistics report
        print("\n[Step 7] Generating statistics report...")
        self.save_statistics_report()
        
        print("\n" + "="*80)
        print("PREPROCESSING ANALYSIS COMPLETE!")
        print(f"All visualizations saved to: {self.figures_path}")
        print(f"Statistics report saved to: {self.stats_path}")
        print(f"Sampled dataset info saved to: {self.stats_path / 'sampled_dataset.csv'}")
        print(f"Split information saved to: {self.stats_path / 'dataset_split.csv'}")
        print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution script for PlantVillage dataset preprocessing
    """
    
    # Configuration - UPDATE THIS PATH
    DATA_PATH = "data/Plant_leave_diseases_dataset_without_augmentation"
    OUTPUT_PATH = "processed_plantvillage"
    TARGET_SIZE = 40000  # Target dataset size after stratified sampling
    
    # Create preprocessor instance
    preprocessor = PlantVillagePreprocessor(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        target_size=TARGET_SIZE
    )
    
    # Run complete analysis
    preprocessor.run_complete_analysis()
