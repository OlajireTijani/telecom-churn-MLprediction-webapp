import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px


def get_clean_data():
    """
    Reads and cleans the input data.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv("data/data.csv")

    # Drop the columns that are not required
    columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude', 
                       'Churn Reason', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Gender', 'Senior Citizen']
    
    # Check if columns exist in the DataFrame before dropping
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    df = df.drop(columns=columns_to_drop)
    
    # Convert numeric columns to numeric types, coercing errors to NaN
    numeric_columns = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def prepare_radar_chart(input_data, df, numeric_columns):
    """
    Creates a radar chart comparing user input to the dataset ranges.

    Args:
        input_data (dict): User input data.
        df (pandas.DataFrame): Dataset for ranges.
        numeric_columns (list): List of numeric columns.

    Returns:
        plotly.graph_objects.Figure: Radar chart figure.
    """
    categories = numeric_columns
    user_values = [input_data[col] if col in input_data else "not a valid input" for col in categories]
    max_values = [df[col].max() for col in categories]
    min_values = [df[col].min() for col in categories]

    fig = go.Figure()

    # Add user input trace
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='User Input'
    ))

    # Add dataset range trace
    fig.add_trace(go.Scatterpolar(
        r=max_values,
        theta=categories,
        fill='none',
        name='Max Values'
    ))

    fig.add_trace(go.Scatterpolar(
        r=min_values,
        theta=categories,
        fill='none',
        name='Min Values'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max_values)]  # Set range dynamically based on dataset
            )
        ),
        showlegend=True,
        title="User Input vs Dataset Ranges"
    )

    return fig


# Streamlit app
def main():
    st.set_page_config(
        page_title="📞 🛰️ Telecom Customer Churn Prediction ML Web App",
        page_icon="📊",
        layout="wide"
    )

    st.title("📞 🛰️ Telecom Customer Churn Prediction ML Web App") 
    st.subheader("This app predicts whether a customer will churn based on various parameters (Designed by Olajire Tijani)")

    df = get_clean_data()
    numeric_columns = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]

    # Create two columns for layout
    col1, col2 = st.columns([1, 4])

    with col1:
        # Customer Input Features
        st.subheader("Customer Input Features".lower())
        input_data = {}
        st.write("### Enter Numeric Values:".lower())
        for col in numeric_columns:
            if col in df.columns:
                input_value = st.text_input(
                    label=f"{col} (Range: {df[col].min()} - {df[col].max()})",
                    value=str(df[col].mean())
                )
                try:
                    input_data[col] = float(input_value)
                except ValueError:
                    st.warning(f"Please enter a valid numeric value for {col}.")
                    input_data[col] = None

        # Prediction Results
        st.subheader("Prediction Results")
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        model = pickle.load(open("model/model.pkl", "rb"))
        if st.button("Predict Churn"):
            input_df = pd.DataFrame([input_data])
            input_array_scaled = scaler.transform(input_df)
            prediction = model.predict(input_array_scaled)
            probas = model.predict_proba(input_array_scaled)

            st.write("Churn Prediction:", "Churn" if prediction[0] == 1 else "Not Churn")
            st.write("Probability of Not Churn:", probas[0][0])
            st.write("Probability of Churn:", probas[0][1])
        
        # Radar Chart
        if input_data:
            st.subheader("Radar Chart of User Input vs Dataset Ranges".lower())
            radar_chart = prepare_radar_chart(input_data, df, numeric_columns)
            st.plotly_chart(radar_chart)

    with col2:
        # Scatter Plot
        st.subheader("Scatter Plot: Monthly Charges vs Total Charges")
        scatter_fig = px.scatter(df, x='Monthly Charges', y='Total Charges', color='Churn Value')
        st.plotly_chart(scatter_fig)

        # Line Chart
        st.subheader("Line Chart: Average Churn Value by Tenure Months")
        line_chart_data = df.groupby('Tenure Months')['Churn Value'].mean()
        st.line_chart(line_chart_data)

    # Footer
    st.markdown("IBM Cognos Analytics 11.1.3+ base samples dataset".upper())
    st.write("Designed and developed by Olajire Tijani".upper())


if __name__ == "__main__":
    main()
