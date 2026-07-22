import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

def clean_and_merge_data(demographics_path, survey_path, output_path):
    # Load data
    demographics = pd.read_csv(demographics_path)
    survey = pd.read_csv(survey_path)
    
    # 1. Inner join on employee_id
    df = pd.merge(demographics, survey, on='employee_id', how='inner')
    
    # 2. Impute missing Likert scale values using Median Imputation
    likert_cols = ['q_work_life_balance', 'q_manager_effectiveness', 'q_career_growth']
    imputer = SimpleImputer(strategy='median')
    df[likert_cols] = imputer.fit_transform(df[likert_cols])
    
    # 3. Replace missing values in open_comment with empty strings
    df['open_comment'] = df['open_comment'].fillna("")
    
    # 4. Feature Engineering: composite_engagement_score
    df['composite_engagement_score'] = df[likert_cols].mean(axis=1)
    
    # 5. Export the output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demographics_path = os.path.join(base_path, 'data', 'raw', 'demographics.csv')
    survey_path = os.path.join(base_path, 'data', 'raw', 'survey_responses.csv')
    output_path = os.path.join(base_path, 'data', 'processed', 'clean_survey_data.csv')
    
    clean_and_merge_data(demographics_path, survey_path, output_path)
