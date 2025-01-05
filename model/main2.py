import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

def create_model(df):
    """
    Creates and trains a Logistic Regression model.

    Args:
        df (pandas.DataFrame): The cleaned DataFrame.

    Returns:
        tuple: A tuple containing the trained model and scaler.
    """
    # Handle missing values by replacing them with a placeholder or dropping them
    df.replace(' ', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Feature selection and preparation
    features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
    x = df[features]
    y = df['Churn Label']

    # One-hot encode categorical features
    categorical_features = [col for col in x.columns if x[col].dtype == object]
    x = pd.get_dummies(x, columns=categorical_features)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))

    return model, scaler

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

def main():
    df = get_clean_data()
    model, scaler = create_model(df)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()