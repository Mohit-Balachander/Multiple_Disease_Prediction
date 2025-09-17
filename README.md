# Medical Multi-Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Overview

This project implements a comparative analysis framework for evaluating machine learning algorithms in medical diagnosis applications. The system provides empirical evidence for algorithm selection in clinical prediction tasks through systematic performance evaluation across multiple pathological conditions.

## Research Objective

The primary objective is to conduct a comprehensive comparative study between conventional and advanced machine learning algorithms for medical diagnosis prediction. This research addresses the fundamental question of whether increased algorithmic complexity consistently translates to improved diagnostic accuracy in healthcare applications.

## Methodology

### Problem Definition

We investigate the performance characteristics of machine learning algorithms across three distinct medical prediction tasks:

1. **Diabetes Mellitus Prediction**: Binary classification for diabetes onset prediction using physiological and demographic parameters
2. **Cardiovascular Disease Detection**: Binary classification for heart disease presence using clinical indicators and diagnostic measurements
3. **Parkinson's Disease Identification**: Binary classification for Parkinson's disease detection through voice analysis parameters

### Algorithm Selection Framework

The study employs a paired comparison methodology, evaluating one conventional algorithm against one advanced algorithm for each medical condition:

| Medical Condition   | Conventional Algorithm | Advanced Algorithm |
| ------------------- | ---------------------- | ------------------ |
| Diabetes            | Logistic Regression    | Random Forest      |
| Heart Disease       | Decision Tree          | Gradient Boosting  |
| Parkinson's Disease | Support Vector Machine | AdaBoost           |

### Evaluation Approach

Performance assessment is conducted using standard binary classification metrics:

- Accuracy: Overall prediction correctness
- Precision: Positive predictive value
- Recall: Sensitivity or true positive rate
- F1-Score: Harmonic mean of precision and recall

All models undergo k-fold cross-validation (k=5) to ensure statistical reliability and minimize overfitting bias.

## System Implementation

The project delivers a comprehensive evaluation platform comprising:

### Core Functionality

1. **Data Preprocessing Pipeline**: Automated data cleaning, feature scaling, and preparation modules for each medical dataset
2. **Model Training Infrastructure**: Standardized training procedures for both conventional and advanced algorithms
3. **Performance Evaluation System**: Automated metric calculation and statistical significance testing
4. **Comparative Analysis Framework**: Side-by-side performance comparison with visualization capabilities
5. **Interactive Web Interface**: Streamlit-based application for real-time prediction and analysis

### Technical Architecture

The system follows a modular architecture enabling:

- Independent algorithm evaluation
- Reproducible experimental conditions
- Scalable addition of new algorithms or datasets
- Comprehensive performance logging and analysis

## Dataset Specifications

### Data Sources

| Dataset               | Origin                             | Samples | Features | Classification Task                  |
| --------------------- | ---------------------------------- | ------- | -------- | ------------------------------------ |
| Pima Indians Diabetes | UCI Machine Learning Repository    | 768     | 8        | Diabetes onset prediction            |
| Heart Disease         | Cleveland Clinic Foundation        | 303     | 13       | Cardiovascular disease detection     |
| Parkinson's Disease   | Oxford Parkinson's Disease Dataset | 195     | 22       | Neurological disorder identification |

### Data Characteristics

Each dataset represents different aspects of medical prediction:

- **Diabetes**: Metabolic health indicators (glucose, BMI, blood pressure)
- **Heart Disease**: Cardiac function parameters (chest pain, cholesterol, ECG results)
- **Parkinson's**: Voice analysis features (frequency variations, amplitude perturbations)

## Expected Outcomes

### Research Contributions

1. **Empirical Evidence**: Quantitative comparison of algorithm performance across diverse medical conditions
2. **Clinical Insights**: Evidence-based recommendations for algorithm selection in healthcare applications
3. **Methodological Framework**: Reusable evaluation methodology for future medical ML studies
4. **Performance Benchmarks**: Established baselines for comparative studies in medical prediction

### Practical Applications

The system enables:

- Healthcare practitioners to make informed algorithm selection decisions
- Researchers to benchmark new algorithms against established methods
- Students to understand comparative machine learning evaluation in medical contexts
- Developers to implement evidence-based medical prediction systems

## Installation

```bash
git clone https://github.com/Mohit-Balachander/medical-ml-diagnosis.git
cd medical-ml-diagnosis
pip install -r requirements.txt
streamlit run app.py
```

## Usage

The system provides an interactive interface for:

1. Selecting algorithm pairs for comparison
2. Inputting patient data for prediction
3. Viewing comparative performance metrics
4. Analyzing prediction results with confidence measures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Mohit Balachander**  
Email: mohitbalachander@gmail.com  
LinkedIn: [Mohit Balachander](https://www.linkedin.com/in/mohit-balachander/)  
GitHub: [Mohit-Balachander](https://github.com/Mohit-Balachander)
