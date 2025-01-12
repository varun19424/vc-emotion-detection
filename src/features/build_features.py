import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def load_params(params_path: str) -> int:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        max_features = params["build_features"]["max_features"]
        if not isinstance(max_features, int) or max_features <= 0:
            raise ValueError(f"Invalid max_features value: {max_features}. Must be a positive integer.")
        logger.info("Parameters loaded successfully.")
        return max_features
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def read_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}.")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading data from {file_path}: {e}")
        raise

def process_text_data(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
    try:
        # Fill missing values
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        logger.info("Missing values filled with empty strings.")

        # Extract features and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Apply Bag of Words (CountVectorizer)
        vectorizer = CountVectorizer(max_features=max_features)

        # Fit and transform the training data
        X_train_bow = vectorizer.fit_transform(X_train)
        logger.info("Bag of Words vectorization applied to training data.")

        # Transform the test data
        X_test_bow = vectorizer.transform(X_test)
        logger.info("Bag of Words vectorization applied to test data.")

        return X_train_bow, y_train, X_test_bow, y_test
    except Exception as e:
        logger.error(f"Error during text processing: {e}")
        raise

def save_data(data_path: str, X_train_bow, y_train, X_test_bow, y_test) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        # Save train data
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        logger.info(f"Training data saved to {data_path}/train_bow.csv.")

        # Save test data
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        logger.info(f"Test data saved to {data_path}/test_bow.csv.")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

def main():
    try:
        # Load parameters
        max_features = load_params('params.yaml')

        # Load data
        train_data = read_data("./data/processed/train_processed.csv")
        test_data = read_data("./data/processed/test_processed.csv")

        # Process text data
        X_train_bow, y_train, X_test_bow, y_test = process_text_data(train_data, test_data, max_features)

        # Save processed data
        data_path = "data/interim"
        save_data(data_path, X_train_bow, y_train, X_test_bow, y_test)

        logger.info("Feature building workflow completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")

if __name__ == "__main__":
    main()
