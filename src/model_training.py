import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

def train_model(data_path, output_dir):
    df = pd.read_csv(data_path)
    
    # Define features and target
    target = 'satisfaction_category'
    
    # Numeric features
    numeric_features = [
        'q_work_life_balance', 'q_manager_effectiveness', 'q_career_growth',
        'composite_engagement_score', 'tenure_years', 'sentiment_score'
    ]
    
    # TF-IDF features
    tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
    
    # Categorical features
    cat_features = ['department', 'role']
    
    # Preprocess categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cats = encoder.fit_transform(df[cat_features])
    encoded_cat_cols = encoder.get_feature_names_out(cat_features)
    
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols)
    
    # Combine features
    X = pd.concat([df[numeric_features], df[tfidf_cols], encoded_df], axis=1)
    y = df[target]
    
    # Handle any remaining NaNs if present (e.g. TFIDF for missing comments was 0)
    X = X.fillna(0)
    
    # Split data (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = rf_model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Save evaluation metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
        
    # Extract feature importance
    importances = rf_model.feature_importances_
    feature_importance_dict = {feat: float(imp) for feat, imp in zip(X.columns, importances)}
    
    # Sort feature importances
    feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    importance_path = os.path.join(output_dir, 'feature_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(feature_importance_dict, f, indent=4)
        
    # Save model
    model_path = os.path.join(output_dir, 'satisfaction_model.pkl')
    # Save a dictionary containing the model and the columns used for training
    joblib.dump({
        'model': rf_model,
        'features': list(X.columns),
        'encoder': encoder
    }, model_path)
    
    print(f"Model and metrics saved to {output_dir}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'processed', 'clean_survey_data.csv')
    train_model(data_path, base_path)
