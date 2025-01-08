import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

dagshub.init(repo_owner='srikanth57-coder', repo_name='latest_mushroom_prediction', mlflow=True)

# Load the data
mlflow.set_experiment("experiment-1")
mlflow.set_tracking_uri("https://dagshub.com/srikanth57-coder/latest_mushroom_prediction.mlflow")


data=pd.read_csv(r"C:\Users\M.Srikanth Reddy\Downloads\mushrooms.csv")
df=pd.DataFrame(data)

df.head()
df.shape
df.describe().T
df.info()
df.isnull().sum()
class1=df['class'].value_counts()

if df.select_dtypes(include=['object']).shape[1] > 0:
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

df.info()

x=df.drop(columns='class',axis=1)
y=df['class']


print(x)
print(y)

scaler= StandardScaler()
X=scaler.fit_transform(x)

X = pd.DataFrame(X, columns=x.columns)
x_train,x_test,y_train,y_test =  train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Start a parent MLflow run to track the overall experiment
with mlflow.start_run(run_name="mushroom prediction Models Experiment"):
    # Iterate over each model in the dictionary
    for model_name, model in models.items():
        # Start a child run within the parent run for each individual model
        with mlflow.start_run(run_name=model_name, nested=True):
            # Train the model on the training data
            model.fit(x_train, y_train)

            # Save the trained model using pickle
            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_filename, "wb"))

            y_pred = model.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1_scr = f1_score(y_test, y_pred)

            # Log metrics with mlflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", pre)
            mlflow.log_metric("recall_score", recall)
            mlflow.log_metric("f1_score", f1_scr)

            # Generate and visualize the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Create a heatmap of the confusion matrix
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plot_filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
            plt.savefig(plot_filename)  # Save the plot
            mlflow.log_artifact(plot_filename)  # Log the confusion matrix plot as an artifact in MLflow

            # Log the model to MLflow
            mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))

            # Log the source code file for reproducibility (the current script)
            if '__file__' in globals():
                mlflow.log_artifact(__file__)
            else:
                print("Running in an interactive environment, skipping source code logging.")
    
    print("All models have been trained and logged as child runs successfully.")