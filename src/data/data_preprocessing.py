import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os 

def load_data(filepath):
    try: 
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        raise

def encode_categorical_columns(data, label_encoder=None):
    try:
        if data.select_dtypes(include=['object']).shape[1] > 0:
            if label_encoder is None:
                label_encoder = LabelEncoder()
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                data[col] = label_encoder.fit_transform(data[col])
        return data, label_encoder
    except Exception as e:
        print(f"Error encoding categorical columns: {e}")
        raise

def scale_numerical_columns(data):
    try:
        # Ensure 'class' column is in the dataset before dropping
        if 'class' not in data.columns:
            raise KeyError("'class' column not found in the data.")
        
        # Separate the features and the target variable
        x = data.drop(columns='class', axis=1)
        y = data['class']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)
        
        # Convert back to DataFrame to retain column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=x.columns)
        
        # Add the target variable back to the DataFrame
        X_scaled_df['class'] = y
        return X_scaled_df
    except Exception as e:
        print(f"Error scaling numerical columns: {e}")
        raise

def save_data(data, filepath):
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        raise

def main():
    try:
        data_raw_path = "./data/raw"
        data_processed_path = "./data/processed"

        # Load raw data
        train_data = load_data(os.path.join(data_raw_path, "train.csv"))
        test_data = load_data(os.path.join(data_raw_path, "test.csv"))
        
        # Encode categorical columns and return the fitted label encoder for future use
        train_processed_data, label_encoder = encode_categorical_columns(train_data)
        test_processed_data, _ = encode_categorical_columns(test_data, label_encoder)

        # Scale numerical columns and return processed data as DataFrame
        train_processed_data = scale_numerical_columns(train_processed_data)
        test_processed_data = scale_numerical_columns(test_processed_data)

        # Create the processed data directory if it doesn't exist
        os.makedirs(data_processed_path, exist_ok=True)

        # Save the processed data
        save_data(train_processed_data, os.path.join(data_processed_path, "train_processed_scaler.csv"))
        save_data(test_processed_data, os.path.join(data_processed_path, "test_processed_scaler.csv"))
    
    except Exception as e:
        print(f"Error in main processing: {e}")

if __name__ == "__main__":
    main()
