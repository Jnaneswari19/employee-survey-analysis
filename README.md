# Employee Survey Analysis Pipeline

This project is an end-to-end data processing, statistical analysis, NLP, machine learning, and dashboard application for analyzing employee survey data.

## Directory Structure
- `data/`: Contains raw generated data and processed datasets.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and statistical testing.
- `src/`: Python source code for data generation, cleaning, NLP processing, and model training.
- `dashboard.py`: Streamlit application for interactive data visualization.
- `hr_recommendations.md`: Executive summary of findings and actionable recommendations.

## Setup Instructions

1. **Install dependencies**:
   Ensure you have Python 3.9+ installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Data Pipeline**:
   Execute the scripts in the following order to generate and process the data:
   ```bash
   python src/data_generator.py
   python src/data_cleaning.py
   python src/nlp_processing.py
   python src/model_training.py
   ```

3. **Run Exploratory Data Analysis**:
   Open and execute the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

4. **Launch the Dashboard**:
   Run the Streamlit application to visualize the results:
   ```bash
   streamlit run dashboard.py
   ```

## Features
- **Data Generation**: Creates realistic synthetic employee demographics and survey responses.
- **Data Preprocessing**: Handles missing values, merges datasets, and engineers composite scores.
- **NLP Processing**: Extracts TF-IDF features and VADER sentiment scores from open-text comments.
- **Machine Learning**: Predicts employee satisfaction categories using a Random Forest Classifier.
- **Interactive Dashboard**: Provides KPIs and interactive visualisations to explore the data.
