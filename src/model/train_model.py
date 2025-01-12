import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from sklearn.ensemble import GradientBoostingClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def load_params(params_path: str):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)["train_model"]
        # Validate parameters
        if params["n_estimators"] <= 0 or params["learning_rate"] <= 0:
            raise ValueError("Invalid parameter values. 'n_estimators' and 'learning_rate' must be positive.")
        logger.info("Parameters loaded successfully.")
        return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing parameter in YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(file_path: str):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading data from {file_path}: {e}")
        raise

def train_model(X_train, y_train, params):
    try:
        # Define and train the Gradient Boosting model
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"], 
            learning_rate=params["learning_rate"]
        )
        clf.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return clf
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model, model_path: str):
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved successfully to {model_path}.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    try:
        # Define file paths
        params_path = 'params.yaml'
        train_data_path = "./data/features/train_bow.csv"
        model_path = 'models/model.pkl'

        # Load parameters
        params = load_params(params_path)

        # Load training data
        train_data = load_data(train_data_path)
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        # Train the model
        clf = train_model(X_train, y_train, params)

        # Save the trained model
        save_model(clf, model_path)

        logger.info("Model training workflow completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")

if __name__ == "__main__":
    main()
