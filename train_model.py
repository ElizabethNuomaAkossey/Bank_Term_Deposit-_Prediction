import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pickle
import os

def train_and_save_model():
    # Load the data
    try:
        df = pd.read_csv('bank-additional-full.csv', sep=';')
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        
        # Print class distribution
        print("\nTarget variable distribution:")
        print(df['y'].value_counts(normalize=True))
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        scale_pos_weight=len(y[y==0])/len(y[y==1])  # Handle class imbalance
    )
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
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    
    # Print detailed evaluation metrics
    print("\nModel Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Calculate F1 scores
    f1_train = f1_score(y_train, model.predict(X_train))
    f1_test = f1_score(y_test, y_pred)
    
    print("\nF1 Scores:")
    print(f"Training F1 Score: {f1_train:.4f}")
    print(f"Testing F1 Score: {f1_test:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    return True

if __name__ == "__main__":
    train_and_save_model()
