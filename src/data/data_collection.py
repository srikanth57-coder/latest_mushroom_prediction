import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
import yaml

def params_load(filepath):

    try:
        with open(filepath,"r") as file:
            params=yaml.safe_load(file)
        return params['data_collection']['test_size']
    except Exception as e:
        print(f"Unexpected error while loading parameters: {e}")
        raise

#test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

def data_load(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        raise

#data=pd.read_csv(r"C:\Users\M.Srikanth Reddy\Downloads\mushrooms.csv")

def split_data(data,test_size):
    try:
        return train_test_split(data,test_size=test_size,random_state=42)
    except Exception as e:
        print(f"Unexpected error during data split: {e}")
        raise

#train_data , test_data = train_test_split(data, test_size=test_size, random_state=42)

def save_data(data,filepath):
    try:
        data.to_csv(filepath,index=False)
    except Exception as e:
        print(f"Unexpected error while saving data to {filepath}: {e}")
        raise
#data_path= os.path.join("data","raw")

def main():
    data_filepath = r"C:\Users\M.Srikanth Reddy\Downloads\mushrooms.csv"
    params_filepath= "params.yaml"
    raw_data_path =  os.path.join("data","raw")

    try:
        data= data_load(data_filepath)
        test_size= params_load(params_filepath)
        train_data,test_data = split_data(data, test_size)

        os.makedirs(raw_data_path)

        save_data(train_data,os.path.join(raw_data_path,"train.csv"))
        save_data(test_data, os.path.join(raw_data_path,"test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred :{e}")

if __name__ == "__main__":
    main()
