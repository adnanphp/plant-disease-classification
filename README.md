#+TITLE: Plant Disease Classification using Deep Learning
#+AUTHOR: Your Name
#+DATE: 2026-03-30
#+LANGUAGE: en
#+OPTIONS: toc:2 num:t
#+STARTUP: content

* Plant Disease Classification using Deep Learning

[[https://www.python.org/][https://img.shields.io/badge/Python-3.9+-blue.svg]]
[[https://pytorch.org/][https://img.shields.io/badge/PyTorch-2.0+-red.svg]]
[[https://www.tensorflow.org/][https://img.shields.io/badge/TensorFlow-2.10+-orange.svg]]
[[https://github.com/features/actions][https://img.shields.io/badge/License-MIT-green.svg]]

A comprehensive deep learning pipeline for plant disease classification using multiple CNN architectures (AlexNet, ResNet-18, MobileNetV2) on the PlantVillage dataset. This project includes data preprocessing, model training, evaluation, and comparative analysis of different architectures.

** Key Features
- Complete preprocessing pipeline for PlantVillage dataset (55,448 → 39,981 balanced images)
- Implementation of three CNN architectures from scratch in PyTorch
- Transfer learning with ResNet50 pretrained on ImageNet
- Comprehensive evaluation with per-class performance analysis
- t-SNE feature visualization
- Automated report generation

** Results at a Glance

#+CAPTION: Model Performance Comparison
| Model               | Test Accuracy | Parameters | Best Class F1 | Worst Class F1 |
|---------------------+---------------+------------+---------------+----------------|
| MobileNetV2         | 97.23%        | 2.27M      | 1.0000        | 0.5714         |
| ResNet-18           | 97.15%        | 11.20M     | 1.0000        | 0.8571         |
| ResNet50 (Fine-tuned) | 96.58%     | 24.70M     | 1.0000        | 0.0000         |
| AlexNet             | 89.86%        | 58.44M     | 0.9960        | 0.6316         |

** Project Structure
#+BEGIN_SRC text
.
├── data/                          # Dataset directory
│   └── processed_plantvillage/    # Preprocessed data
├── preprocessing.py               # Data preprocessing script
├── a_alexnet_pytorch.py          # AlexNet training (PyTorch)
├── ab_resnet18_pytorch.py        # ResNet-18 training (PyTorch)
├── ac_mobilenetv2_pytorch.py     # MobileNetV2 training (PyTorch)
├── ada_resnet50_tl.py            # Transfer learning (ResNet50)
├── requirements.txt              # Python dependencies
├── results/                      # All experiment results
│   ├── alexnet_results/
│   ├── resnet18_results/
│   ├── mobilenetv2_results/
│   └── transfer_learning_results/
└── README.org                    # This file
#+END_SRC

** Dataset
The PlantVillage dataset [[https://arxiv.org/abs/1511.08060][Hughes & Salathé, 2015]] contains 55,448 images of healthy and diseased plant leaves across 39 classes.

*** Dataset Statistics
#+CAPTION: Dataset Split
| Split       | Number of Images | Percentage |
|-------------+------------------+------------|
| Training    | 27,986           | 70.0%      |
| Validation  | 5,997            | 15.0%      |
| Test        | 5,998            | 15.0%      |
|-------------+------------------+------------|
| **Total**   | **39,981**       | **100%**   |

*** Sample Images
#+CAPTION: Sample images from the PlantVillage dataset
[[./figures/sample_images.png]]

** Installation

*** Prerequisites
- Python 3.9 or higher
- pip package manager

*** Setup
#+BEGIN_SRC bash
# Clone the repository
git clone https://github.com/adnanphp/plant-disease-classification.git
cd plant-disease-classification

# Install dependencies
pip install -r requirements.txt
#+END_SRC

*** Requirements
#+BEGIN_SRC text
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.64.0
Pillow>=9.0.0
#+END_SRC

** Usage

*** Step 1: Preprocess the Dataset
#+BEGIN_SRC bash
python preprocessing.py
#+END_SRC
This script:
- Loads raw PlantVillage images
- Performs stratified sampling (target: 40,000 images)
- Splits data into train/validation/test (70/15/15)
- Saves preprocessed data to =processed_plantvillage/=

*** Step 2: Train Models

**** Train AlexNet
#+BEGIN_SRC bash
python a_alexnet_pytorch.py
#+END_SRC

**** Train ResNet-18
#+BEGIN_SRC bash
python ab_resnet18_pytorch.py
#+END_SRC

**** Train MobileNetV2
#+BEGIN_SRC bash
python ac_mobilenetv2_pytorch.py
#+END_SRC

**** Transfer Learning with ResNet50
#+BEGIN_SRC bash
# Initial training (frozen base)
python ada_resnet50_tl_initial.py

# Fine-tuning
python ada_resnet50_tl_finetune.py

# Generate report
python ada_z_generate_finetune_report.py
#+END_SRC

** Results

*** Training Curves

#+CAPTION: ResNet-18 Training Progress
[[./figures/resnet18_training.png]]

#+CAPTION: MobileNetV2 Training Progress
[[./figures/mobilenetv2_training.png]]

*** Per-Class Performance Analysis

**** ResNet-18
#+CAPTION: Best Performing Classes (ResNet-18)
| Class                              | Precision | Recall | F1-Score |
|------------------------------------+-----------+--------+----------|
| Corn___Common_rust                 | 1.0000    | 1.0000 | 1.0000   |
| Squash___Powdery_mildew            | 0.9950    | 1.0000 | 0.9975   |
| Corn___healthy                     | 0.9921    | 1.0000 | 0.9960   |
| Orange___Haunglongbing             | 0.9917    | 0.9983 | 0.9950   |
| Grape___Esca_(Black_Measles)       | 0.9868    | 1.0000 | 0.9934   |

#+CAPTION: Worst Performing Classes (ResNet-18)
| Class                                    | Precision | Recall | F1-Score |
|------------------------------------------+-----------+--------+----------|
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.8571  | 0.8571 | 0.8571   |
| Tomato___Target_Spot                     | 0.8632    | 0.8632 | 0.8632   |
| Potato___healthy                         | 0.9091    | 0.9091 | 0.9091   |
| Corn___Northern_Leaf_Blight              | 0.9109    | 0.9109 | 0.9109   |
| Tomato___Early_blight                    | 0.9189    | 0.9189 | 0.9189   |

**** MobileNetV2
#+CAPTION: Best Performing Classes (MobileNetV2)
| Class                              | Precision | Recall | F1-Score |
|------------------------------------+-----------+--------+----------|
| Apple___Cedar_apple_rust           | 1.0000    | 1.0000 | 1.0000   |
| Tomato___healthy                   | 1.0000    | 1.0000 | 1.0000   |
| Orange___Haunglongbing             | 0.9983    | 0.9983 | 0.9983   |
| Corn___Common_rust                 | 0.9922    | 0.9922 | 0.9922   |
| Corn___healthy                     | 0.9844    | 1.0000 | 0.9921   |

#+CAPTION: Worst Performing Classes (MobileNetV2)
| Class                                    | Precision | Recall | F1-Score |
|------------------------------------------+-----------+--------+----------|
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.5714  | 0.5714 | 0.5714   |
| Potato___healthy                         | 0.8571    | 0.8571 | 0.8571   |
| Corn___Northern_Leaf_Blight              | 0.8642    | 0.8642 | 0.8642   |
| Tomato___Early_blight                    | 0.8835    | 0.8835 | 0.8835   |
| Grape___Black_rot                        | 0.9378    | 0.9378 | 0.9378   |

*** Confusion Matrices

#+CAPTION: ResNet-18 Confusion Matrix
[[./figures/resnet18_confusion.png]]

#+CAPTION: MobileNetV2 Confusion Matrix
[[./figures/mobilenetv2_confusion.png]]

*** Feature Visualization (t-SNE)

#+CAPTION: ResNet-18 Feature Space
[[./figures/resnet18_tsne.png]]

#+CAPTION: MobileNetV2 Feature Space
[[./figures/mobilenetv2_tsne.png]]

** Transfer Learning Results

*** Performance Improvement
#+CAPTION: Transfer Learning Comparison
| Metric          | Initial Model | Fine-tuned Model | Improvement |
|-----------------+---------------+------------------+-------------|
| Test Accuracy   | 94.48%        | 96.58%           | +2.10%      |
| Macro Precision | 0.9196        | 0.9410           | +0.0214     |
| Macro Recall    | 0.8892        | 0.9217           | +0.0325     |
| Macro F1-Score  | 0.8984        | 0.9286           | +0.0302     |

*** Fine-tuning Progress
#+CAPTION: Fine-tuning Training Curves
[[./figures/finetune_training.png]]

** Key Findings

1. **MobileNetV2 is the most efficient model**
   - Highest accuracy (97.23%) with only 2.27M parameters
   - 26× smaller than AlexNet, 5× smaller than ResNet-18
   - Ideal for mobile/edge deployment

2. **ResNet-18 provides the most balanced performance**
   - 97.15% accuracy with consistent per-class results
   - Best performance on challenging corn disease classes
   - No class failures (worst F1 = 0.8571)

3. **Transfer learning shows limitations**
   - Complete failure on Potato___healthy class (F1 = 0.0000)
   - Fine-tuning improved overall accuracy but didn't fix class-specific failures
   - From-scratch training more reliable for this dataset

4. **PyTorch implementations outperform TensorFlow**
   - Same ResNet-18 architecture: 97.15% (PyTorch) vs 87.40% (TensorFlow)
   - Framework choice significantly impacts training stability

** Model Recommendations

| Deployment Scenario                    | Recommended Model | Reason                                 |
|----------------------------------------+-------------------+----------------------------------------|
| Mobile/Edge devices                    | MobileNetV2       | Smallest (2.27M), highest accuracy     |
| Agricultural research                  | ResNet-18         | Balanced, best on corn diseases        |
| Production (unconstrained resources)   | ResNet-18         | Consistent across all classes          |
| Transfer learning                      | NOT Recommended   | Class-specific failures                |

** Citation

If you use this code or findings in your research, please cite:

#+BEGIN_SRC bibtex
@article{hughes2015plantvillage,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
#+END_SRC

** License

This project is licensed under the MIT License - see the [[LICENSE]] file for details.

** Acknowledgments

- PlantVillage dataset creators (Hughes & Salathé, 2015)
- PyTorch and TensorFlow teams for excellent deep learning frameworks
- The open-source community for invaluable tools and libraries

** Contact

- Author: Your Name
- Email: your.email@example.com
- GitHub: [[https://github.com/adnanphp/plant-disease-classification][@adnanphp]]

** References

- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
- Krizhevsky, A., et al. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
- Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*.

---

#+BEGIN_CENTER
**Star this repository if you found it useful!** ⭐
#+END_CENTER
