import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import dagshub
import mlflow
from mlflow.models import infer_signature
import seaborn as sns
import matplotlib.pyplot as plt

dagshub.init(repo_owner='srikanth57-coder', repo_name='latest_mushroom_prediction', mlflow=True)

# Load the data
mlflow.set_experiment("best with dvc pipeline")
mlflow.set_tracking_uri("https://dagshub.com/srikanth57-coder/latest_mushroom_prediction.mlflow")

def load_data(datapath):
    try:
        return pd.read_csv(datapath)
    except FileNotFoundError:
        print(f"Error: The file {datapath} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {datapath} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: The file {datapath} could not be parsed.")
        raise
    except Exception as e:
        print(f"Unexpected error loading data from {datapath}: {e}")
        raise

def split_data(data):
    try:
        x = data.drop(columns=['class'], axis=1)
        y = data['class']
        return x, y
    except KeyError:
        print("Error: 'class' column not found in the data.")
        raise
    except Exception as e:
        print(f"Unexpected error during data splitting: {e}")
        raise

def load_model(filepath):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        print(f"Error: The model file {filepath} was not found.")
        raise
    except pickle.UnpicklingError:
        print(f"Error: The model file {filepath} could not be unpickled.")
        raise
    except Exception as e:
        print(f"Unexpected error loading model from {filepath}: {e}")
        raise

def predict_model(model, x_test, y_test):
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        criterion = params["model_training"]["criterion"]
        max_depth = params["model_training"]["max_depth"]
        min_samples_leaf = params["model_training"]["min_samples_leaf"]
        min_samples_split = params["model_training"]["min_samples_split"]

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_scr = f1_score(y_test, y_pred)
        # Log the metrics
        mlflow.log_param("Test_size", test_size)
        mlflow.log_param("criterion",criterion)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("min_samples_leaf",min_samples_leaf)
        mlflow.log_param("min_samples_split",min_samples_split)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", pre)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_scr)

        model_name = "Best Model"

         # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        
        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)
        
        # Log the model
        #mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))

        metrics_dict = {
            'accuracy': acc,
            'precision': pre,
            'recall': recall,
            'f1_score': f1_scr
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")
    

def save_metrics(metrics,metrics_path) :
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}: {e}")
    
def main():
    try:
        test_data_path = "./data/processed/test_processed_scaler.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        model_name = "Best Model"

        test_data = load_data(test_data_path)
        X_test, y_test = split_data(test_data)
        model = load_model(model_path)

        # Start MLflow run
        with mlflow.start_run() as run:
            metrics = predict_model(model, X_test, y_test)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            
            # Log the source code file
            mlflow.log_artifact(__file__)

            signature = infer_signature(X_test,model.predict(X_test))

            mlflow.sklearn.log_model(model,"Best Model",signature=signature)

            #Save run ID and model info to JSON File
            run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)

    except Exception as e:
        raise Exception(f"An Error occurred: {e}")

if __name__ == "__main__":
    main()
