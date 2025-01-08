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
from sklearn.tree import DecisionTreeClassifier

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

clf = DecisionTreeClassifier()

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:
    grid_search.fit(x_train, y_train)

    # Log all hyperparameter combinations
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination {i+1}", nested=True) as child_run:
            params = grid_search.cv_results_['params'][i]
            mean_test_score = grid_search.cv_results_['mean_test_score'][i]

            # Log the parameters and their corresponding score
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", mean_test_score)

    print("Best parameters found: ", grid_search.best_params_)

    best_rfc = grid_search.best_estimator_
    best_rfc.fit(x_train, y_train)

    # Save the model with pickle
    pickle.dump(best_rfc, open("model.pkl", "wb"))

    model = pickle.load(open('model.pkl',"rb"))

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate metrics
    acc= accuracy_score(y_test,y_pred)
    pre= precision_score(y_test,y_pred)
    recall= recall_score(y_test,y_pred)
    f1_scr= f1_score(y_test,y_pred)

    # Log metrics with mlflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", pre)
    mlflow.log_metric("recall_score",recall)
    mlflow.log_metric("f1_score",f1_scr)


    # Log model parameters
    mlflow.log_param("best_params", grid_search.best_params_)
            
    df1 = mlflow.data.from_pandas(data)
            
    cm= confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sb.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matric")

    plt.savefig("confusion_matrix.png")

    mlflow.log_input(df1)

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(grid_search.best_estimator_,"best_model")

    mlflow.log_artifact(__file__)

    print("acc",acc)
    print("pre",pre)
    print("recall",recall)
    print("f1_src",f1_scr)