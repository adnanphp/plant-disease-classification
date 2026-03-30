# ResNet-18 Training Report - PlantVillage Dataset (PyTorch)

**Generated:** 2026-03-24 08:20:41

## Dataset Information
- **Number of Classes:** 39
- **Training Samples:** 27986
- **Validation Samples:** 5997
- **Test Samples:** 5998

## Model Architecture
- **Architecture:** ResNet-18 (He et al., 2016)
- **Input Size:** (224, 224)
- **Total Parameters:** 11.20M
- **Trainable Parameters:** 11.20M
- **Framework:** PyTorch
- **Device:** cpu

## Training Configuration
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Max Epochs:** 15
- **Early Stopping Patience:** 15
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss
- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=7, factor=0.5)

### Data Augmentation
- **Random Horizontal Flip:** Yes
- **Random Affine:** Yes (translation ±10%)
- **Color Jitter:** Yes (brightness ±20%, contrast ±20%)
- **Normalization:** ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Results
- **Test Accuracy:** 0.9715 (97.15%)
- **Macro Precision:** 0.9648
- **Macro Recall:** 0.9636
- **Macro F1-Score:** 0.9636
- **Training Time:** 489.25 minutes
- **Best Validation Accuracy:** 97.00% (epoch 14)

## Per-Class Performance (Top 10 by F1-Score)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Corn___Common_rust | 1.0000 | 1.0000 | 1.0000 | 129.0 |
| Squash___Powdery_mildew | 0.9950 | 1.0000 | 0.9975 | 199.0 |
| Corn___healthy | 0.9921 | 1.0000 | 0.9960 | 126.0 |
| Orange___Haunglongbing_(Citrus... | 0.9917 | 0.9983 | 0.9950 | 596.0 |
| Grape___Esca_(Black_Measles) | 0.9868 | 1.0000 | 0.9934 | 150.0 |
| Apple___Black_rot | 1.0000 | 0.9851 | 0.9925 | 67.0 |
| Strawberry___Leaf_scorch | 0.9836 | 1.0000 | 0.9917 | 120.0 |
| Grape___Leaf_blight_(Isariopsi... | 1.0000 | 0.9828 | 0.9913 | 116.0 |
| Tomato___Tomato_Yellow_Leaf_Cu... | 0.9880 | 0.9931 | 0.9905 | 580.0 |
| Strawberry___healthy | 0.9800 | 1.0000 | 0.9899 | 49.0 |
| ... and 29 more classes | ... | ... | ... | ... |

### Best Performing Classes (Top 5)
- **Corn___Common_rust**: F1-Score = 1.0000
- **Squash___Powdery_mildew**: F1-Score = 0.9975
- **Corn___healthy**: F1-Score = 0.9960
- **Orange___Haunglongbing_(Citrus_greening)**: F1-Score = 0.9950
- **Grape___Esca_(Black_Measles)**: F1-Score = 0.9934

### Worst Performing Classes (Bottom 5)
- **Corn___Cercospora_leaf_spot Gray_leaf_spot**: F1-Score = 0.8571
- **Tomato___Target_Spot**: F1-Score = 0.8632
- **Potato___healthy**: F1-Score = 0.9091
- **Corn___Northern_Leaf_Blight**: F1-Score = 0.9109
- **Tomato___Early_blight**: F1-Score = 0.9189

## Key Observations
- **Excellent Performance:** ResNet-18 achieved 97.15% accuracy on the test set
- **Training Stability:** The model showed consistent learning with validation accuracy improving over epochs
- **Efficiency:** ResNet-18 with 11.2M parameters provides good accuracy-to-parameter ratio
- **PyTorch Implementation:** Successfully implemented ResNet-18 from scratch using PyTorch framework

## Recommendations

Based on the analysis:
1. **ResNet-18 is effective** for plant disease classification on this dataset
2. **Consider deeper ResNet variants** (e.g., ResNet-50) if higher accuracy is required
3. **Transfer learning** could potentially improve performance further
4. **Model size (11.2M parameters)** is efficient for deployment