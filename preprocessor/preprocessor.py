import pandas as pd
import numpy as np

class Preprocessor:
    """
    A class for preprocessing data by handling missing values and creating time-based features.
    """
    
    def __init__(self):
        """
        Initialize the Preprocessor class.
        """
        self.numerical_cols = None
        self.feature_columns = None
    
    def find_missing_hours(self, data, plants, providers):
        """
        Finds missing hours within the range of `effectivedate` for each plant and provider.

        Args:
            data (pd.DataFrame): The plant forecast DataFrame.
            plants (list): List of plants to include in the analysis.
            providers (list): List of providers to check.

        Returns:
            dict: A dictionary where keys are plant names, and values are DataFrames of missing hours for each provider.
        """
        # Filter data for the specified plants and providers
        filtered_data = data[
            (data['plant_name'].isin(plants)) & (data['forecast_provider'].isin(providers))
        ]

        # Extract hour from the datetime column
        filtered_data['hour'] = filtered_data['effectivedate'].dt.floor('h')

        # Initialize result dictionary
        missing_hours = {}

        # Loop through each plant
        for plant in plants:
            plant_data = filtered_data[filtered_data['plant_name'] == plant]
            plant_missing = {}

            # Loop through each provider
            for provider in providers:
                provider_data = plant_data[plant_data['forecast_provider'] == provider]
                if not provider_data.empty:
                    # Determine the full range of hours
                    full_range = pd.date_range(
                        start=provider_data['hour'].min(),
                        end=provider_data['hour'].max(),
                        freq='h'
                    )

                    # Find missing hours
                    recorded_hours = pd.to_datetime(provider_data['hour'].unique())
                    missing = set(full_range) - set(recorded_hours)
                    plant_missing[provider] = sorted(missing)

            if plant_missing:
                missing_hours[plant] = plant_missing

        return missing_hours
    
    def create_missing_rows(self, missing_hours_dict, original_data, feature_columns):
        """
        Creates a DataFrame with missing rows filled with NaNs based on missing_hours_dict.
        
        Args:
            missing_hours_dict (dict): Output from find_missing_hours function.
            original_data (pd.DataFrame): The original DataFrame.
            feature_columns (list): List of feature columns to include.
        
        Returns:
            pd.DataFrame: DataFrame containing all missing rows.
        """
        if not missing_hours_dict:
            # Return an empty DataFrame with the same columns as original_data
            return pd.DataFrame(columns=original_data.columns)
            
        missing_rows = []
        
        for plant, providers_missing in missing_hours_dict.items():
            for provider, missing_hours in providers_missing.items():
                for timestamp in missing_hours:
                    row = {
                        'plant_name': plant,
                        'forecast_provider': provider,
                        'effectivedate': timestamp
                    }
                    # Initialize all feature columns with NaN
                    for col in feature_columns:
                        row[col] = pd.NA
                    missing_rows.append(row)
        
        if not missing_rows:
            # Return an empty DataFrame with the same columns as original_data
            return pd.DataFrame(columns=original_data.columns)
            
        # Create DataFrame with the same dtypes as original_data
        missing_df = pd.DataFrame(missing_rows)
        
        # Ensure dtypes match with original_data where possible
        for col in missing_df.columns:
            if col in original_data.columns:
                try:
                    missing_df[col] = missing_df[col].astype(original_data[col].dtype)
                except:
                    # If conversion fails, keep the original dtype
                    pass
        
        return missing_df
    
    def interpolate_group(self, group):
        """
        Performs time-based interpolation on a group of data.
        
        Args:
            group (pd.DataFrame): A group of data (usually for a specific plant and provider).
            
        Returns:
            pd.DataFrame: The group with interpolated values.
        """
        # Set 'effectivedate' as the index
        group = group.set_index('effectivedate')
        
        # Perform time-based interpolation
        group[self.numerical_cols] = group[self.numerical_cols].interpolate(method='time')
        
        # Reset the index to turn 'effectivedate' back into a column
        group = group.reset_index()
        
        return group
    
    def preprocess(self, data, providers=['Provider 1', 'Provider 2'], forecast_columns=['fc0', 'fc1200', 'fc40', 'fc55', 'fc60', 'fc75']):
        """
        Preprocess the data by handling missing values and creating time features.
        
        Args:
            data (pd.DataFrame): The input data.
            providers (list): List of providers to include.
            forecast_columns (list): List of forecast columns to reshape.
            
        Returns:
            pd.DataFrame: The preprocessed data.
        """
        # Make a copy of the input data to avoid modifying the original
        df = data.copy()
        
        # Ensure 'effectivedate' is in datetime format
        df['effectivedate'] = pd.to_datetime(df['effectivedate'])
        
        # Define feature columns (all columns except plant_name, forecast_provider, effectivedate)
        self.feature_columns = [col for col in df.columns if col not in ['plant_name', 'forecast_provider', 'effectivedate']]
        
        # Clip production values to be non-negative
        if 'production' in df.columns:
            df['production'] = df['production'].clip(lower=0)
        
        # Find missing hours
        plants = df['plant_name'].unique().tolist()
        missing_hours_dict = self.find_missing_hours(df, plants, providers)
        
        # Create rows for missing hours
        missing_df = self.create_missing_rows(missing_hours_dict, df, self.feature_columns)
        
        # Combine original and missing data
        if missing_df.empty:
            complete_df = df.copy()
        else:
            complete_df = pd.concat([df, missing_df], ignore_index=True)
        
        # Sort by plant, provider, and timestamp
        complete_df.sort_values(['plant_name', 'forecast_provider', 'effectivedate'], inplace=True)
        
        # Reset index
        complete_df.reset_index(drop=True, inplace=True)
        
        # Define numerical columns to impute
        self.numerical_cols = [col for col in self.feature_columns if complete_df[col].dtype in ['float64', 'int64']]
        
        # Apply interpolation within each group
        complete_df = complete_df.groupby(['plant_name', 'forecast_provider']).apply(self.interpolate_group)
        
        # After groupby.apply, the index might become a MultiIndex. Reset it.
        complete_df.reset_index(drop=True, inplace=True)
        
        # Apply forward fill and backward fill within each group for any remaining NaNs
        complete_df[self.numerical_cols] = complete_df.groupby(['plant_name', 'forecast_provider'])[self.numerical_cols].transform(lambda group: group.ffill().bfill())
        
        # Check for categorical columns with missing values
        categorical_cols = complete_df.select_dtypes(include=['object', 'category']).columns.tolist()
        missing_categorical = complete_df[categorical_cols].isnull().sum()
        
        # Drop categorical columns with missing values (typically sunrise and sunset)
        cols_to_drop = missing_categorical[missing_categorical > 0].index.tolist()
        if cols_to_drop:
            complete_df.drop(cols_to_drop, axis=1, inplace=True)
        
        # Check if forecast columns exist in the dataframe
        available_forecast_cols = [col for col in forecast_columns if col in complete_df.columns]
        
        if available_forecast_cols:
            # Melt the dataframe to unpivot forecast columns
            melted = complete_df.melt(
                id_vars=[col for col in complete_df.columns if col not in available_forecast_cols],
                value_vars=available_forecast_cols,
                var_name='forecast_type',
                value_name='forecast_value'
            )
            
            # Create a combined column for forecast_provider and forecast_type
            melted['provider_forecast'] = (
                melted['forecast_provider'].str.replace(' ', '').str.lower() + '_' + melted['forecast_type']
            )
            
            # Pivot the table to reshape, keeping all other columns intact
            reshaped = melted.pivot_table(
                index=[col for col in complete_df.columns if col not in available_forecast_cols + ['forecast_provider']],
                columns='provider_forecast',
                values='forecast_value'
            ).reset_index()
            
            # Flatten the columns after pivoting
            reshaped.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in reshaped.columns]

            # drop nan rows
            reshaped = reshaped.dropna()

            # remove duplicates on effectivedate and plant_name
            reshaped = reshaped.drop_duplicates(subset=['effectivedate', 'plant_name'])
            
            return reshaped
        
        return complete_df
    
