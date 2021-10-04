from src.utils.all_utils import read_yaml, create_directory, save_reports
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
import joblib

def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2
    
def evaluate(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
     
    # Save file in the local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]['artifacts_dir'] 
    split_data_dir = config['artifacts']['split_data_dir']
    
    # Get the path of train and test csv files
    test_data_filename = config['artifacts']['test']
    
    # Get the full path for the train data csv file
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename) 
    
    # Read the train.csv files    
    test_data = pd.read_csv(test_data_path) 
    
    #Split of train_X and train_Y
    test_X = test_data.drop("quality", axis = 1)
    test_Y = test_data['quality']
   
    # Get the model path
    model_dir = config['artifacts']['model_dir'] 
    model_filename = config['artifacts']['model_filename']
    model_path = os.path.join(artifacts_dir, model_dir, model_filename)
    
    # Load the model to the variable lr
    lr = joblib.load(model_path) 
     
    # Fit/train the model 
    predicted_values = lr.predict(test_X)
    
    #Calculate the rmse, mae and the r2 values
    rmse, mae, r2 = evaluate_metrics(test_Y, predicted_values)

    #Store the evaluation metrics in a separate directory
    scores = {
            "rmse":rmse,
            "mae": mae,
            "r2" :r2 
    }
    
    scores_dir = config['artifacts']['reports_dir']
    scores_filename = config['artifacts']['scores']
    
    scores_dir_path = os.path.join(artifacts_dir, scores_dir)
    create_directory([scores_dir_path])
    scores_filepath = os.path.join(scores_dir_path, scores_filename)
    
    save_reports(report=scores, report_path=scores_filepath) 
     

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml") 
    
    parsed_args = args.parse_args()
    
    evaluate(config_path = parsed_args.config, params_path=parsed_args.params)