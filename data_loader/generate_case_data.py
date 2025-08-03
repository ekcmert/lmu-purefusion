import os
import pandas as pd
import sys
from pathlib import Path

# Import directly from files
from data_loader import DataLoader
from feature_engineer.feature_engineer import FeatureEngineer
from constants.cases import CASE_DIC

def generate_case_datasets(output_dir="data/case_datasets"):
    """
    Generate datasets for each case in CASE_DIC by loading, preprocessing,
    and applying feature engineering, then save them to the specified directory.
    
    Args:
        output_dir (str): Directory to save the case datasets
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize DataLoader
    data_loader = DataLoader(data_path="data/fc_data.csv")
    
    # Process each case
    for case_id, case_config in CASE_DIC.items():
        print(f"Processing Case {case_id}...")
        
        # Extract configuration
        plants = case_config.get("plants", [])
        providers = case_config.get("providers", [])
        features = case_config.get("features", [])
        case_type = case_config.get("type", "")
        
        # Format provider names for feature engineer (lowercase, no spaces)
        formatted_providers = [p.replace(" ", "").lower() for p in providers]
        print(providers)
        print(plants)
        # Load and preprocess data
        processed_data = data_loader.process_data(plants=plants, providers=providers)
        
        if processed_data is None or processed_data.empty:
            print(f"No data available for Case {case_id}. Skipping.")
            continue
        
        # Apply feature engineering
        feature_engineer = FeatureEngineer(providers=formatted_providers)
        engineered_data = feature_engineer.fit_transform(processed_data)
        
        if engineered_data is None or engineered_data.empty:
            print(f"Feature engineering failed for Case {case_id}. Skipping.")
            continue
        
        # Ensure all required features are in the dataset
        # Some features from CASE_DIC might not be in the dataset after processing
        available_features = [feat for feat in features if feat in engineered_data.columns]
        missing_features = [feat for feat in features if feat not in engineered_data.columns]
        
        if missing_features:
            print(f"Warning for Case {case_id}: The following features are missing: {missing_features}")
        
        # Select features in the order specified in CASE_DIC
        # Include 'plant_name' and 'effectivedate' as they are key identifier columns
        key_columns = ['plant_name', 'effectivedate']
        
        # Add features that aren't already in key_columns to maintain order
        selected_columns = key_columns.copy()
        for feat in features:
            if feat in engineered_data.columns and feat not in selected_columns:
                selected_columns.append(feat)
        
        # Ensure we're not trying to select columns that don't exist
        final_columns = [col for col in selected_columns if col in engineered_data.columns]
        
        # Select columns in specified order
        case_data = engineered_data[final_columns]
        
        # Save the dataset
        output_file = os.path.join(output_dir, f"case_{case_id}_{case_type.replace(' ', '_').lower()}.csv")
        case_data.to_csv(output_file, index=False)
        
        print(f"Case {case_id} saved to {output_file}")
        print(f"Dataset shape: {case_data.shape}")
        print(f"Features: {', '.join(case_data.columns.tolist())}")
        print("-" * 80)

if __name__ == "__main__":
    generate_case_datasets()
    print("All case datasets generated successfully!")
