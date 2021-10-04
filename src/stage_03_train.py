from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
import os
from sklearn.linear_model import ElasticNet
import joblib

def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
     
    # Save file in the local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]['artifacts_dir'] 
    split_data_dir = config['artifacts']['split_data_dir']
    
    # Get the path of train and test csv files
    train_data_filename = config['artifacts']['train']
    
    # Get the full path for the train data csv file
    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename) 
    
    # Read the train.csv files    
    train_data = pd.read_csv(train_data_path) 
    
    #Split of train_X and train_Y
    train_X = train_data.drop("quality", axis = 1)
    train_Y = train_data['quality']
   
    # Define the parameters over here
    alpha_parameter = params['model_params']['ElasticNet']['alpha']
    ratio = params['model_params']['ElasticNet']['l1_ratio']
    random_state = params['model_params']['ElasticNet']['random_state']
   
    # Define the Elastic Net model here
    lr = ElasticNet(alpha=alpha_parameter, l1_ratio=ratio, random_state=random_state)
    
    # Fit/train the model 
    lr.fit(train_X,train_Y)
    
    #Path for model directory
    model_dir = config['artifacts']['model_dir']
    model_filename = config['artifacts']['model_filename']
    model_dir = os.path.join(artifacts_dir, model_dir)
    
    #Create the directory
    create_directory([model_dir])
    
    #Get the model path
    model_path = os.path.join(model_dir, model_filename)
    
    #Save the model to the directory
    joblib.dump(lr, model_path)   

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml") 
    
    parsed_args = args.parse_args()
    
    train(config_path = parsed_args.config, params_path=parsed_args.params)