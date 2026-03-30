# AlexNet Training Report - PlantVillage Dataset (PyTorch)

**Generated:** 2026-03-25 13:39:44

## Dataset Information
- **Number of Classes:** 39
- **Test Accuracy:** 0.8986 (89.86%)

## Model Architecture
- **Architecture:** AlexNet (Krizhevsky et al., 2012)
- **Total Parameters:** 58.44M
- **Trainable Parameters:** 58.44M

## Results Summary

### Overall Metrics
- **Test Accuracy:** 0.8986 (89.86%)
- **Macro Avg Precision:** 0.8773
- **Macro Avg Recall:** 0.8766
- **Macro Avg F1-Score:** 0.8721
- **Training Time:** 7.50 minutes

### Per-Class Performance (Top 10 by F1-Score)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Corn___healthy | 0.9921 | 1.0000 | 0.9960 | 126.0 |
| Squash___Powdery_mildew | 0.9659 | 0.9950 | 0.9802 | 199.0 |
| Orange___Haunglongbing_(Citrus... | 0.9750 | 0.9815 | 0.9783 | 596.0 |
| Tomato___Tomato_Yellow_Leaf_Cu... | 0.9894 | 0.9638 | 0.9764 | 580.0 |
| Corn___Common_rust | 0.9549 | 0.9845 | 0.9695 | 129.0 |
| Soybean___healthy | 0.9526 | 0.9855 | 0.9688 | 551.0 |
| Grape___Leaf_blight_(Isariopsi... | 0.9820 | 0.9397 | 0.9604 | 116.0 |
| Background_without_leaves | 0.9587 | 0.9431 | 0.9508 | 123.0 |
| Apple___Cedar_apple_rust | 0.9643 | 0.9310 | 0.9474 | 29.0 |
| Potato___Early_blight | 0.9898 | 0.8981 | 0.9417 | 108.0 |
| ... and 29 more classes | ... | ... | ... | ... |

### Best Performing Classes (Top 5)
- **Corn___healthy**: F1-Score = 0.9960
- **Squash___Powdery_mildew**: F1-Score = 0.9802
- **Orange___Haunglongbing_(Citrus_greening)**: F1-Score = 0.9783
- **Tomato___Tomato_Yellow_Leaf_Curl_Virus**: F1-Score = 0.9764
- **Corn___Common_rust**: F1-Score = 0.9695

### Worst Performing Classes (Bottom 5)
- **Corn___Cercospora_leaf_spot Gray_leaf_spot**: F1-Score = 0.6316
- **Tomato___Early_blight**: F1-Score = 0.7005
- **Apple___Apple_scab**: F1-Score = 0.7006
- **Potato___healthy**: F1-Score = 0.7179
- **Tomato___Target_Spot**: F1-Score = 0.7206

## Key Observations
- **Good Performance:** AlexNet achieved 89.86% accuracy with moderate training time.
- **Model Size:** 58.4M parameters - moderate and suitable for deployment

## Recommendations

Based on the analysis:
1. **AlexNet is effective** for plant disease classification on this dataset
2. **Consider transfer learning** if higher accuracy is required (e.g., using pretrained ResNet or EfficientNet)
3. **Model size (58.4M parameters)** is suitable for deployment on standard hardware