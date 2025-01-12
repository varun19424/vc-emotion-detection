import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            clf = pickle.load(file)
        logger.info("Model loaded successfully.")
        return clf
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
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

def evaluate_model(clf, X_test, y_test):
    try:
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
        }
        logger.info("Model evaluation metrics calculated successfully.")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, output_path: str):
    try:
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Metrics saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

def main():
    try:
        # Define paths
        model_path = 'models/model.pkl'
        test_data_path = './data/features/test_bow.csv'
        metrics_path = 'reports/metrices.json'

        # Load the trained model
        clf = load_model(model_path)

        # Load test data
        test_data = load_data(test_data_path)
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate model
        metrics = evaluate_model(clf, X_test, y_test)

        # Save metrics
        save_metrics(metrics, metrics_path)

        logger.info("Model evaluation completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")

if __name__ == "__main__":
    main()
