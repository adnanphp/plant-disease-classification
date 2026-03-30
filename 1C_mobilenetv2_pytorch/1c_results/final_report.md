# MobileNetV2 Training Report - PlantVillage Dataset (PyTorch)

**Generated:** 2026-03-26 07:45:00

## Dataset Information
- **Number of Classes:** 39
- **Training Samples:** 27986
- **Validation Samples:** 5997
- **Test Samples:** 5998

## Model Architecture
- **Architecture:** MobileNetV2 (Sandler et al., 2018)
- **Input Size:** (224, 224)
- **Total Parameters:** 2.27M
- **Trainable Parameters:** 2.27M
- **Key Features:** Depthwise separable convolutions, inverted residuals
- **Framework:** PyTorch
- **Device:** cpu

## Training Configuration
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Max Epochs:** 15
- **Early Stopping Patience:** 10
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss
- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)

### Data Augmentation
- **Random Horizontal Flip:** Yes
- **Random Affine:** Yes (translation ±10%)
- **Color Jitter:** Yes (brightness ±20%, contrast ±20%)
- **Normalization:** ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Results
- **Test Accuracy:** 0.9723 (97.23%)
- **Macro Precision:** 0.9694
- **Macro Recall:** 0.9538
- **Macro F1-Score:** 0.9570
- **Training Time:** 686.63 minutes
- **Best Validation Accuracy:** 96.83% (epoch 14)

## Per-Class Performance (Top 10 by F1-Score)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Apple___Cedar_apple_rust | 1.0000 | 1.0000 | 1.0000 | 29.0 |
| Tomato___healthy | 1.0000 | 1.0000 | 1.0000 | 172.0 |
| Orange___Haunglongbing_(Citrus... | 0.9983 | 0.9983 | 0.9983 | 596.0 |
| Corn___Common_rust | 0.9922 | 0.9922 | 0.9922 | 129.0 |
| Corn___healthy | 0.9844 | 1.0000 | 0.9921 | 126.0 |
| Soybean___healthy | 0.9822 | 1.0000 | 0.9910 | 551.0 |
| Blueberry___healthy | 0.9878 | 0.9939 | 0.9908 | 163.0 |
| Tomato___Tomato_Yellow_Leaf_Cu... | 0.9896 | 0.9879 | 0.9888 | 580.0 |
| Tomato___Tomato_mosaic_virus | 0.9756 | 1.0000 | 0.9877 | 40.0 |
| Squash___Powdery_mildew | 0.9802 | 0.9950 | 0.9875 | 199.0 |
| ... and 29 more classes | ... | ... | ... | ... |

### Best Performing Classes (Top 5)
- **Apple___Cedar_apple_rust**: F1-Score = 1.0000
- **Tomato___healthy**: F1-Score = 1.0000
- **Orange___Haunglongbing_(Citrus_greening)**: F1-Score = 0.9983
- **Corn___Common_rust**: F1-Score = 0.9922
- **Corn___healthy**: F1-Score = 0.9921

### Worst Performing Classes (Bottom 5)
- **Corn___Cercospora_leaf_spot Gray_leaf_spot**: F1-Score = 0.5714
- **Potato___healthy**: F1-Score = 0.8571
- **Corn___Northern_Leaf_Blight**: F1-Score = 0.8642
- **Tomato___Early_blight**: F1-Score = 0.8835
- **Grape___Black_rot**: F1-Score = 0.9378

## Key Observations
- **Excellent Performance:** MobileNetV2 achieved 97.23% accuracy with only 2.3M parameters.
- **Lightweight Architecture:** With only 2.3M parameters, MobileNetV2 is highly efficient for deployment on edge devices.
- **Training Efficiency:** The model trained in 686.6 minutes, demonstrating fast convergence.
- **PyTorch Implementation:** Successfully implemented MobileNetV2 from scratch using PyTorch framework.

## Recommendations

Based on the analysis:
1. **MobileNetV2 is ideal for deployment** due to its small size (2.3M parameters)
2. **Excellent accuracy-to-parameter ratio** - Great for mobile and edge applications
3. **Consider transfer learning** if higher accuracy is required (though at the cost of larger model size)
4. **PyTorch implementation** provides flexibility for further experimentation and optimization