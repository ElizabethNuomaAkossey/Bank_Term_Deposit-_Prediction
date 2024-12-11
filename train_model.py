import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_and_save_model():
    # Load the data
    try:
        df = pd.read_csv('bank-additional-full.csv', sep=';')
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset not found. Please ensure the dataset is in the correct location.")
        return False

    # Print initial data info
    print("\nInitial data info:")
    print(df.dtypes)

    # Prepare features
    # Drop less important features
    columns_to_drop = ['default', 'contact', 'pdays', 'poutcome']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Initialize dictionary to store encoders
    encoder_dict = {}

    # Convert categorical variables to category type first
    categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'month', 'day_of_week']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Encode categorical variables
    for col in categorical_columns:
        if col in df.columns:
            print(f"\nEncoding {col}...")
            print(f"Unique values before encoding: {df[col].unique()}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoder_dict[col] = le
            print(f"Unique values after encoding: {df[col].unique()}")

    # Encode target variable
    print("\nEncoding target variable...")
    le_target = LabelEncoder()
    df['y'] = le_target.fit_transform(df['y'])

    # Ensure numeric columns are properly handled
    numeric_columns = ['age', 'balance', 'duration', 'campaign', 'previous', 
                      'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                      'euribor3m', 'nr.employed']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"\nFixed missing values in {col} with median: {median_value}")

    # Split features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # Print final data info
    print("\nFinal data info before training:")
    print(X.dtypes)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("\nTraining the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the encoders
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoder_dict, f)

    print("\nModel and encoders have been trained and saved successfully!")
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"Training Score: {train_score:.4f}")
    print(f"Testing Score: {test_score:.4f}")
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("\nFeature names saved successfully!")
    print(f"Features used: {feature_names}")
    
    return True

if __name__ == "__main__":
    train_and_save_model()
