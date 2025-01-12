import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['make_dataset']['test_size']
        if not (0 < test_size < 1):
            raise ValueError(f"Test size {test_size} is out of the valid range (0, 1).")
        logger.info("Parameters loaded successfully.")
        return test_size
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.info("Data loaded successfully from URL.")
        return df
    except Exception as e:
        logger.error(f"Error reading data from URL: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting data processing.")
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 0, 'sadness': 1})
        logger.info("Data processing complete.")
        return final_df
    except KeyError as e:
        logger.error(f"Missing expected column: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.info(f"Data saved successfully to {data_path}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        # Load parameters
        test_size = load_params('params.yaml')
        
        # Read dataset
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        
        # Process dataset
        final_df = process_data(df)
        
        # Train-test split
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Save processed data
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        
        logger.info("Workflow completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")

if __name__ == "__main__":
    main()
