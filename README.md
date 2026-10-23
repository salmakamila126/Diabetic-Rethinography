# CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging

📌 Project Overview

Diabetic Retinopathy (DR) is a leading cause of vision impairment globally. Accurate multi-stage classification is vital for early clinical intervention. Standard Convolutional Neural Networks (CNNs) often struggle with long-range spatial context, while standard Vision Transformers (ViTs) can overlook local fine-grained microvascular lesion features.

This repository provides an automated deep learning pipeline based on the original research:
- **Original Paper**: [CAD-ViT: Coordinate Attention-Enhanced Vision Transformer With Dilated Feature Fusion for Diabetic Retinopathy Staging](https://ieeexplore.ieee.org/document/11009024) (IEEE Access).
- **Authors**: YE WANG, XIAOFANG GOU, AND WENMAN LI

## 📊 Datasets Used

The pipeline utilizes two primary fundus photography datasets for training and validation:

| Dataset | Description | Source Link |
| :--- | :--- | :--- |
| **APTOS 2019** | Kaggle Blindness Detection competition dataset containing 3,662 labeled high-resolution fundus images across 5 severity stages. | [Kaggle ](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |
| **DDR (Diabetic Retinopathy)** | Multi-center dataset with 13,673 fundus images featuring varied lighting, field of view, and camera hardware. | [Kaggle ](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset) |


