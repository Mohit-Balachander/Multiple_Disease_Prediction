# 🏥 Multiple Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> A comprehensive comparative analysis of conventional versus advanced machine learning algorithms for multi-disease prediction across three distinct medical conditions.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Diseases & Algorithms](#-diseases--algorithms)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [Authors](#-authors)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact & Support](#-contact--support)

## 🎯 Overview

Medical diagnosis represents a critical domain where accurate prediction models can significantly impact patient outcomes and healthcare delivery efficiency. This project presents a comprehensive comparative analysis of conventional versus advanced machine learning algorithms for multi-disease prediction.

### 🔬 Research Objectives

- Compare conventional vs. advanced ML algorithms in medical diagnosis
- Evaluate performance across three distinct medical conditions
- Challenge traditional algorithmic assumptions in healthcare AI
- Provide evidence-based guidance for clinical deployment

## ✨ Features

- **🔍 Multi-Disease Prediction**: Diabetes, Heart Disease, and Parkinson's Disease
- **📊 Interactive Dashboard**: Real-time model performance comparison
- **🎛️ User-Friendly Interface**: Streamlit-based web application
- **📈 Comprehensive Analytics**: Confusion matrices, performance metrics, and visualizations
- **🔄 Model Comparison**: Side-by-side evaluation of conventional vs. advanced algorithms
- **💡 Clinical Recommendations**: Evidence-based algorithm selection guidance

## 🧬 Diseases & Algorithms

| Disease                    | Conventional Algorithm | Advanced Algorithm | Dataset                 |
| -------------------------- | ---------------------- | ------------------ | ----------------------- |
| **Diabetes** 🩺            | Logistic Regression    | Random Forest      | Pima Indians Diabetes   |
| **Heart Disease** ❤️       | Decision Tree          | Gradient Boosting  | UCI Heart Disease       |
| **Parkinson's Disease** 🧠 | Support Vector Machine | AdaBoost           | Voice Analysis Features |

## 🚀 Installation

### Prerequisites

```bash
Python 3.7+
pip package manager
```

### Quick Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Mohit-Balachander/medical-ml-diagnosis.git
   cd medical-ml-diagnosis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
plotly>=5.15.0
streamlit-option-menu>=0.3.2
pickle-mixin>=1.0.2
```

## 📁 Project Structure

```
medical-ml-diagnosis/
├── 📄 app.py                      # Main Streamlit application
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Project documentation
├── 📄 LICENSE                     # MIT License
├── 📂 saved_models/               # Trained model files
│   ├── 📂 old/                    # Conventional algorithms
│   │   ├── diabetes_model.sav
│   │   ├── heart_disease_model.sav
│   │   ├── parkinsons_model.sav
│   │   └── parkinsons_scaler.sav
│   └── 📂 new/                    # Advanced algorithms
│       ├── diabetes_model.sav
│       ├── diabetes_scaler.sav
│       ├── heart_disease_model.sav
│       ├── heart_scaler.sav
│       └── parkinsons_model.sav
├── 📂 dataset/                    # Medical datasets
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
├── 📂 notebooks/                  # Jupyter notebooks
    ├── diabetes_analysis.ipynb
    ├── heart_disease_analysis.ipynb
    └── parkinsons_analysis.ipynb

```

## 🎮 Usage

### 1. Model Comparison Dashboard

Access comprehensive performance metrics and visualizations:

```
Navigate to "Model Comparison Dashboard" in the sidebar
```

### 2. Disease Prediction

Make individual predictions for each condition:

**Diabetes Prediction:**

- Input: Age, BMI, Glucose levels, etc.
- Models: Logistic Regression vs Random Forest

**Heart Disease Prediction:**

- Input: Age, Chest Pain Type, Blood Pressure, etc.
- Models: Decision Tree vs Gradient Boosting

**Parkinson's Disease Prediction:**

- Input: Voice analysis parameters (22 features)
- Models: SVM vs AdaBoost

### 3. Model Version Selection

Choose between conventional ("old") and advanced ("new") algorithms using the sidebar selector.

## 📊 Datasets

### Diabetes Dataset (Pima Indians)

- **Size**: 768 samples, 8 features
- **Features**: Pregnancies, Glucose, Blood Pressure, BMI, etc.
- **Target**: Binary (Diabetic/Non-diabetic)

### Heart Disease Dataset (UCI)

- **Size**: 303 samples, 13 features
- **Features**: Age, Sex, Chest Pain Type, Cholesterol, etc.
- **Target**: Binary (Disease/No Disease)

### Parkinson's Dataset

- **Size**: 195 samples, 22 features
- **Features**: Voice frequency parameters, jitter, shimmer
- **Target**: Binary (Parkinson's/Healthy)

## 📈 Model Performance

### Key Metrics Evaluated

- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall

### Performance Summary

| Disease       | Conventional      | Advanced        | Best Performer         |
| ------------- | ----------------- | --------------- | ---------------------- |
| Diabetes      | **LogReg: 77.3%** | RF: 76.0%       | Logistic Regression    |
| Heart Disease | DT: 75.4%         | **GB: 83.6%**   | Gradient Boosting      |
| Parkinson's   | **SVM: 94.9%**    | AdaBoost: 89.7% | Support Vector Machine |

## 🔍 Key Findings

### 🏆 Surprising Results

1. **Conventional Algorithms Excel**: Traditional methods outperformed advanced algorithms in 2/3 cases
2. **Perfect Recall**: SVM achieved 100% recall for Parkinson's disease detection
3. **Diabetes Accuracy**: Logistic Regression showed 1.3% higher accuracy than Random Forest
4. **Heart Disease**: Only category where advanced algorithm (Gradient Boosting) significantly outperformed conventional method

### 💡 Clinical Implications

- **Algorithm Selection**: Domain-specific evaluation is crucial
- **Interpretability**: Conventional algorithms often provide better clinical interpretability
- **Performance vs Complexity**: More complex doesn't always mean better in medical applications

## 📸 Screenshots

### Dashboard Overview

![Diabetes Dashboard](/Images/diabetes_prediction.jpg)
![Heart Dashboard](/Images/heart_disease_prediction.jpg)
![Parkinsons Dashboard](/Images/parkinsons_prediction.jpg)

### Model Comparison

![Accuracy Comparison](/Images/Accuracy_Comparison.png)
![F1 Score Comparison](/Images/F1%20Score%20Comparison.png)
![Precision Comparison](/Images/Precision_Comparison.png)
![Recall Comparison](/Images/Recall_Comparison.png)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### 📋 Contributing Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass

<!-- ## 🔬 Research Paper

This project is based on our research paper:

> "Comparative Analysis of Conventional versus Advanced Machine Learning Algorithms for Multi-Disease Prediction"

**Abstract**: Medical diagnosis represents a critical domain where accurate prediction models can significantly impact patient outcomes and healthcare delivery efficiency...
-->

## 👥 Authors

- **Mohit Balachander** - _Lead Researcher_ - [Mohit-Balachander](https://github.com/Mohit-Balachander)

## 📜 License

This project is licensed under the [MIT License](LICENSE) – feel free to use, modify, and distribute this software with proper attribution.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing datasets
- Pima Indians Diabetes Database contributors
- Voice analysis research community
- Open source machine learning community

## 📞 Contact & Support

- **Email**: mohitbalachander@gmail.com
- **LinkedIn**: [Mohit Balachander](https://www.linkedin.com/in/mohit-balachander/)
- **Issues**: [GitHub Issues](https://github.com/Mohit-Balachander/medical-ml-diagnosis/issues)

---

<div align="center">
  <sub>Built with ❤️ for advancing healthcare through AI</sub>
</div>
