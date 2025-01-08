import pandas as pd
import numpy as np 
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
import yaml

def params_load(filepath):
    try:
        with open(filepath,"r") as file:
            params=yaml.safe_load(file)
            return params['model_training']['criterion'], params['model_training']['max_depth'], params['model_training']['min_samples_leaf'], params['model_training']['min_samples_split']
    except FileNotFoundError:
        print(f"Error: The parameter file {filepath} was not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error reading the YAML file {filepath}: {e}")
        raise
    except KeyError as e:
        print(f"Error: Missing expected key {e} in the YAML file.")
        raise
    except Exception as e:
        print(f"Unexpected error loading parameters from {filepath}: {e}")
        raise
def load_data(data_path):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The data file {data_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The data file {data_path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: The data file {data_path} could not be parsed.")
        raise
    except Exception as e:
        print(f"Unexpected error loading data from {data_path}: {e}")
        raise
#train_data=pd.read_csv("./data/processed/train_processed.csv")
def split_data(data):
    try:
        x= data.drop(columns=['class'],axis=1)
        y= data['class']
        return x,y
    except KeyError:
        print("Error: 'class' column not found in the data.")
        raise
    except Exception as e:
        print(f"Unexpected error during data splitting: {e}")
        raise

def  train_model(x,y,criterion, max_depth, min_samples_leaf, min_samples_split):
    try:
        dtc = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split)
        return dtc.fit(x,y)
    except ValueError as e:
        print(f"Error training the model: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during model training: {e}")
        raise

def save_model(model,filepath):
    try:
        with open(filepath,"wb") as file:
            pickle.dump(model,file)
    except Exception as e:
        print(f"Error saving the model to {filepath}: {e}")
        raise

def main():
    try:
        data_path = "./data/processed/train_processed_scaler.csv"
        params_filepath = "params.yaml"
        model_name = "models/model.pkl"

        criterion, max_depth, min_samples_leaf, min_samples_split = params_load(params_filepath)

        train_data = load_data(data_path)
        x_train,y_train = split_data(train_data)

        model = train_model(x_train,y_train,criterion,max_depth,min_samples_leaf,min_samples_split)
        save_model(model,model_name)
    except Exception as e:
        print(f"Error in main processing: {e}")

if __name__ == "__main__":
    main()





#x_train = train_data.iloc[:, 1:].values
#y_train = train_data.iloc[:, 0].values 

#dtc =  DecisionTreeClassifier(criterion = 'gini', max_depth = 10, min_samples_leaf = 1, min_samples_split = 2)
#dtc.fit(x_train,y_train)

#pickle.dump(dtc,open("model.pkl","wb"))