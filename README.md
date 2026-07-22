# Employee Survey Analysis Pipeline 📊

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

An end-to-end HR analytics pipeline using Python and Streamlit to predict employee satisfaction through Natural Language Processing (NLP) sentiment analysis and Machine Learning. 

This project simulates, processes, and analyzes employee survey data to extract actionable insights for HR leadership, featuring statistical testing and a predictive Random Forest model.

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Insights & Recommendations](#-insights--recommendations)

## 📖 Overview
Employee retention and satisfaction are critical metrics for organizational success. This project provides a robust, data-driven methodology to analyze employee sentiment by moving beyond simple averages. It utilizes machine learning to identify hidden drivers of employee satisfaction and NLP to extract contextual sentiment from open-text feedback.

## ✨ Key Features
- **Synthetic Data Generation**: Simulates realistic, biased employee demographics and Likert-scale survey responses.
- **Robust Preprocessing**: Handles missing data via median imputation and engineers composite engagement scores.
- **Statistical Analysis**: Implements non-parametric hypothesis testing (Kruskal-Wallis H-test) to identify significant variances across departments.
- **NLP Sentiment Extraction**: Uses NLTK's VADER lexicon and TF-IDF vectorization to quantify the emotional tone of open-ended employee comments.
- **Machine Learning Classification**: Trains a `RandomForestClassifier` (with balanced class weights) to predict overall employee satisfaction categories and extract feature importance.
- **Interactive Dashboard**: A fully responsive Streamlit web application providing dynamic KPI tracking and Plotly visualizations.

## 🏗 Project Architecture
1. **Data Ingestion** `(src/data_generator.py)` -> Generates raw CSVs.
2. **Data Cleaning** `(src/data_cleaning.py)` -> Cleans and merges datasets.
3. **NLP Processing** `(src/nlp_processing.py)` -> Computes sentiment scores and TF-IDF features.
4. **Model Training** `(src/model_training.py)` -> Trains ML model and outputs `satisfaction_model.pkl` & evaluation metrics.
5. **Visualization** `(dashboard.py)` -> Consumes processed data for the Streamlit UI.

## 📂 Repository Structure
```text
employee-survey-analysis/
├── data/
│   ├── raw/                  # Generated synthetic datasets
│   └── processed/            # Cleaned data with NLP & composite features
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # EDA & Statistical testing
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # Data synthesis script
│   ├── data_cleaning.py      # Preprocessing script
│   ├── nlp_processing.py     # NLTK text processing script
│   └── model_training.py     # Random Forest training script
├── dashboard.py              # Streamlit interactive application
├── hr_recommendations.md     # Executive summary & HR action plan
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/employee-survey-analysis.git
cd employee-survey-analysis
```

2. **Create a virtual environment (Optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage Guide

### 1. Run the Data Pipeline
Execute the processing scripts in sequence to generate data, clean it, extract NLP features, and train the model:
```bash
python src/data_generator.py
python src/data_cleaning.py
python src/nlp_processing.py
python src/model_training.py
```

### 2. Launch the Interactive Dashboard
Spin up the local Streamlit server to interact with the visualizations:
```bash
streamlit run dashboard.py
```
*The dashboard will automatically open in your default web browser at `http://localhost:8501`.*

### 3. Explore the Jupyter Notebook
For a deep dive into the exploratory data analysis and statistical testing:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 📈 Insights & Recommendations
Please refer to the `hr_recommendations.md` file for a complete executive summary of the model's findings, including key drivers of satisfaction and targeted interventions for HR leadership.

---
*Developed as a demonstration of applied Data Science & Machine Learning in Human Resources.*
