import os
import pandas as pd
import sys
import traceback

# Add parent directory to path to import Preprocessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessor.preprocessor import Preprocessor
from feature_engineer.feature_engineer import FeatureEngineer

class DataLoader:
    """
    A class for loading and preprocessing forecast data.
    Applies filtering by plant and provider and handles preprocessing via the Preprocessor class.
    """
    
    def __init__(self, data_path="data/fc_data.csv"):
        """
        Initialize the DataLoader class.
        
        Args:
            data_path (str): Path to the CSV data file.
        """
        self.data_path = data_path
        self.preprocessor = Preprocessor()
        self.data = None
        
    def load_data(self):
        """
        Load the data from the CSV file.
        
        Returns:
            pd.DataFrame: The loaded data.
        """
        try:
            df = pd.read_csv(self.data_path, parse_dates=['effectivedate'])
            self.data = df
            return df
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            traceback.print_exc()
            return None
            
    def filter_data(self, plants=None, providers=None):
        """
        Filter the data by plants and providers.
        
        Args:
            plants (list): List of plants to include. If None, include all plants.
            providers (list): List of providers to include. If None, include all providers.
            
        Returns:
            pd.DataFrame: The filtered data.
        """
        if self.data is None:
            self.load_data()
            
        if self.data is None:
            return None
            
        filtered_data = self.data.copy()
        
        if plants is not None:
            filtered_data = filtered_data[filtered_data['plant_name'].isin(plants)]
            
        if providers is not None:
            filtered_data = filtered_data[filtered_data['forecast_provider'].isin(providers)]
            
        return filtered_data
        
    def process_data(self, plants=None, providers=None):
        """
        Load, filter, and preprocess the data.
        
        Args:
            plants (list): List of plants to include. If None, include all plants.
            providers (list): List of providers to include. If None, include all providers.
            
        Returns:
            pd.DataFrame: The processed data.
        """
        filtered_data = self.filter_data(plants, providers)
        
        if filtered_data is None or filtered_data.empty:
            print("No data to process after filtering.")
            return None
            
        # Define providers list for preprocessing
        preprocessor_providers = providers if providers is not None else filtered_data['forecast_provider'].unique().tolist()
        
        # Apply preprocessing
        processed_data = self.preprocessor.preprocess(filtered_data, providers=preprocessor_providers)
        
        return processed_data

# Example usage
if __name__ == "__main__":
    # Example usage of the DataLoader class
    plants = ["Plant A", "Plant B"]
    providers = ["Provider 1", "Provider 2"]
    
    loader = DataLoader()
    
    # Load and process data for specific plants and providers
    processed_data = loader.process_data(plants=plants, providers=providers)
    
    if processed_data is not None:
        # Apply feature engineering
        feature_engineer = FeatureEngineer(providers=[p.replace(" ", "").lower() for p in providers])
        engineered_data = feature_engineer.fit_transform(processed_data)
        
        print(f"Preprocessed data shape: {processed_data.shape}")
        print(f"Engineered data shape: {engineered_data.shape}")
        print(f"Unique plants: {engineered_data['plant_name'].unique()}")
        print(f"Sample engineered data:\n{engineered_data.head()}")
    else:
        print("Failed to process data.")
