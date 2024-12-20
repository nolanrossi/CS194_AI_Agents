# TeleAEye: AI Eye Disease Diagnosis with Vision Transformers

## Overview
TeleAEye is a groundbreaking AI-driven system designed to enhance eye disease diagnostics by integrating affordable smartphone-compatible fundus camera technology with advanced machine learning models, such as Vision Transformers and fine-tuned large language models (LLMs). This project aims to democratize access to ophthalmic care by offering accurate, explainable, and cost-effective diagnostic tools.

This repository contains an implementation of the methodologies outlined in the research paper **"TeleAEye: AI Eye Disease Diagnosis with a Novel Smartphone Fundus Camera and Vision Transformers"**.

Diagnosis probabilites were produced for the eye.jpg from the eye disease diagnosis models from the google collab notbooke

## Setup

### Prerequisites
- Python 3.8 or above
- Jupyter Notebook
- Libraries:
  - `tensorflow`
  - `torch`
  - `transformers`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `opencv-python`
  - `yolov5`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/teleaeye.git
   cd teleaeye
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

## Usage

### 1. Preprocessing Fundus Images
- Load fundus images using the data preprocessing module.
- Perform augmentation and normalization to prepare the data for training.

### 2. Training Vision Transformers
- Use pre-trained Vision Transformers (`vit-base-patch16-224-in21k`) and fine-tune them on the dataset.
- Apply Multi-Step Transfer Learning (MSTL) for enhanced diagnostic accuracy.

### 3. Model Evaluation
- Evaluate the models using metrics such as AUROC, sensitivity, and specificity.
- Compare results against baseline models to validate improvements.

### 4. Multimodal RAG Chatbot
- Integrate diagnostic outputs with the Retrieval-Augmented Generation (RAG) pipeline.
- Provide explainable diagnostic insights by combining LLM-generated context with model predictions.

## Results
- **Diagnostic Accuracy**: Achieved AUROC of 96.2% across six prevalent ocular diseases.
- **Cost Efficiency**: System hardware cost reduced to $6.80 per unit, enabling widespread adoption.
- **Explainability**: Enhanced clinician trust through contextual outputs from the RAG chatbot.

## Data
- **Dataset**: Fundus images with labeled eye diseases.
- **Preprocessing**: Augmentation techniques include rotation, scaling, and cropping.

## Acknowledgments
This project is supported by the University of California, Berkeley, and was made possible by the contributions of:
- Tien-Lan Sun
- Anaiy Somalwar
- Dhruv Kulkarni
- Nolan Rossi
- Alvin Xiao
- Jyoti Rani

