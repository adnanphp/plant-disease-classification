# ResNet50 Initial Model Report - PlantVillage Dataset (PyTorch)

**Generated:** 2026-03-25 16:37:16

## Dataset Information
- **Number of Classes:** 39
- **Training Samples:** 27986
- **Validation Samples:** 5997
- **Test Samples:** 5998

## Model Architecture
- **Base Model:** ResNet50 (pretrained on ImageNet, frozen)
- **Custom Head:** Dense(512) + BatchNorm + ReLU + Dropout(0.5) → Dense(256) + BatchNorm + ReLU + Dropout(0.3) → Dense(39)
- **Total Parameters:** 24.70M
- **Trainable Parameters:** 1.19M
- **Framework:** PyTorch
- **Device:** CPU

## Training Configuration
- **Batch Size:** 32
- **Learning Rate:** 0.00001
- **Max Epochs:** 15
- **Early Stopping Patience:** 10
- **Input Size:** (224, 224)
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss

## Results
- **Test Accuracy:** 0.9448 (94.48%)
- **Macro Precision:** 0.9196
- **Macro Recall:** 0.8892
- **Macro F1-Score:** 0.8984
- **Best Validation Accuracy:** 95.13% (epoch 15)

## Best Performing Classes (Top 5)
- **Grape___Leaf_blight_(Isariopsis_Leaf_Spot)**: F1-Score = 1.0000
- **Grape___healthy**: F1-Score = 1.0000
- **Corn___Common_rust**: F1-Score = 0.9961
- **Strawberry___Leaf_scorch**: F1-Score = 0.9958
- **Squash___Powdery_mildew**: F1-Score = 0.9950

## Worst Performing Classes (Bottom 5)
- **Potato___healthy**: F1-Score = 0.0000
- **Tomato___Tomato_mosaic_virus**: F1-Score = 0.6207
- **Tomato___Early_blight**: F1-Score = 0.6437
- **Apple___Cedar_apple_rust**: F1-Score = 0.7111
- **Corn___Cercospora_leaf_spot Gray_leaf_spot**: F1-Score = 0.7921

## Key Observations
- **Strong Performance:** ResNet50 with frozen base achieved 94.48% test accuracy
- **Efficient Training:** Only 1.19M parameters trained, reaching 95.13% validation accuracy
- **Good Generalization:** Small gap between training and validation accuracy
- **Ready for Fine-tuning:** Model provides a strong baseline for fine-tuning