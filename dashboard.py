import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Employee Satisfaction Dashboard", layout="wide")

@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'processed', 'clean_survey_data.csv')
    if not os.path.exists(data_path):
        return pd.DataFrame() # Return empty df if not run yet
    return pd.read_csv(data_path)

df = load_data()

if df.empty:
    st.error("Data not found. Please run the data pipeline first.")
    st.stop()

st.title("Employee Satisfaction Dashboard")

# Sidebar Filters
st.sidebar.header("Filters")

# Department Filter
departments = df['department'].unique().tolist()
selected_depts = st.sidebar.multiselect("Select Department", departments, default=departments)

# Role Filter
roles = df['role'].unique().tolist()
selected_roles = st.sidebar.multiselect("Select Role", roles, default=roles)

# Apply filters
filtered_df = df[
    (df['department'].isin(selected_depts)) & 
    (df['role'].isin(selected_roles))
]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# KPI Cards
col1, col2, col3 = st.columns(3)

with col1:
    avg_satisfaction = filtered_df['overall_satisfaction'].mean()
    st.metric(label="Avg Overall Satisfaction", value=f"{avg_satisfaction:.2f}")

with col2:
    respondent_count = len(filtered_df)
    st.metric(label="Respondent Count", value=f"{respondent_count}")

with col3:
    if 'sentiment_score' in filtered_df.columns:
        mean_sentiment = filtered_df['sentiment_score'].mean()
        st.metric(label="Mean Sentiment Score", value=f"{mean_sentiment:.2f}")
    else:
        st.metric(label="Mean Sentiment Score", value="N/A")

st.markdown("---")

# Visualizations
col4, col5 = st.columns(2)

with col4:
    st.subheader("Satisfaction by Department")
    fig1 = px.box(filtered_df, x='department', y='overall_satisfaction', color='department',
                  title="Overall Satisfaction Distribution by Department")
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    st.subheader("Sentiment vs Satisfaction")
    if 'sentiment_score' in filtered_df.columns:
        fig2 = px.scatter(filtered_df, x='sentiment_score', y='overall_satisfaction',
                          color='satisfaction_category', hover_data=['employee_id', 'department'],
                          title="Sentiment Score vs Overall Satisfaction")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sentiment score data not available. Please run NLP processing.")
