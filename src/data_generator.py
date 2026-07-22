import pandas as pd
import numpy as np
import os

def generate_data(num_employees=1000):
    np.random.seed(42)
    
    departments = ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance']
    roles = ['Individual Contributor', 'Manager', 'Director']
    
    emp_ids = [f"EMP{str(i).zfill(4)}" for i in range(1, num_employees + 1)]
    dept_choices = np.random.choice(departments, size=num_employees, p=[0.25, 0.35, 0.1, 0.15, 0.15])
    role_choices = np.random.choice(roles, size=num_employees, p=[0.7, 0.2, 0.1])
    tenure_years = np.random.uniform(0.5, 20.0, size=num_employees)
    
    demographics = pd.DataFrame({
        'employee_id': emp_ids,
        'department': dept_choices,
        'role': role_choices,
        'tenure_years': np.round(tenure_years, 1)
    })
    
    # Survey metrics
    q_manager = np.random.randint(1, 6, size=num_employees)
    q_career = np.random.randint(1, 6, size=num_employees)
    
    # Work life balance with bias for engineering
    q_wlb = []
    for dept in dept_choices:
        if dept == 'Engineering':
            # Lower WLB for Engineering
            score = int(np.round(np.random.normal(2.5, 1.0)))
        else:
            score = int(np.round(np.random.normal(3.8, 1.0)))
        score = max(1, min(5, score)) # clip to 1-5
        q_wlb.append(score)
    
    q_wlb = np.array(q_wlb)
    
    # Overall satisfaction
    # Weighted average + noise
    weighted_score = (q_wlb * 0.4) + (q_manager * 0.3) + (q_career * 0.3)
    noise = np.random.normal(0, 0.3, size=num_employees)
    overall_satisfaction = np.clip(weighted_score + noise, 1, 5)
    
    # Satisfaction categories
    satisfaction_category = []
    for val in overall_satisfaction:
        if val <= 2.5:
            satisfaction_category.append('Low')
        elif val <= 3.5:
            satisfaction_category.append('Medium')
        else:
            satisfaction_category.append('High')
            
    # Open comments
    comments_low = [
        "Overworked and underpaid.", 
        "Management doesn't listen.", 
        "Need better work life balance.", 
        "Not enough growth opportunities.", 
        "Poor communication."
    ]
    comments_med = [
        "It's okay here.", 
        "Some good days, some bad days.", 
        "Benefits are fine, but work is repetitive.", 
        "Average place to work.", 
        "Decent team."
    ]
    comments_high = [
        "Great place to work!", 
        "Love my team.", 
        "Excellent work life balance and growth.", 
        "Very supportive management.", 
        "Best company I've worked for!"
    ]
    
    open_comments = []
    for cat in satisfaction_category:
        if cat == 'Low':
            open_comments.append(np.random.choice(comments_low))
        elif cat == 'Medium':
            open_comments.append(np.random.choice(comments_med))
        else:
            open_comments.append(np.random.choice(comments_high))
            
    survey_data = pd.DataFrame({
        'employee_id': emp_ids,
        'q_work_life_balance': q_wlb.astype(float),
        'q_manager_effectiveness': q_manager.astype(float),
        'q_career_growth': q_career.astype(float),
        'overall_satisfaction': overall_satisfaction,
        'satisfaction_category': satisfaction_category,
        'open_comment': open_comments
    })
    
    # Introduce ~5% missing values in likert columns
    likert_cols = ['q_work_life_balance', 'q_manager_effectiveness', 'q_career_growth']
    for col in likert_cols:
        mask = np.random.rand(num_employees) < 0.05
        survey_data.loc[mask, col] = np.nan
        
    return demographics, survey_data

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_path, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    demographics, survey = generate_data()
    
    demo_path = os.path.join(raw_dir, 'demographics.csv')
    survey_path = os.path.join(raw_dir, 'survey_responses.csv')
    
    demographics.to_csv(demo_path, index=False)
    survey.to_csv(survey_path, index=False)
    
    print(f"Generated synthetic data and saved to {raw_dir}")
