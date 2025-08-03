import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, providers=None):
        """
        Initialize the FeatureEngineer
        
        Args:
            providers (list): List of provider names (e.g. ['provider1', 'provider2'])
        """
        self.providers = providers if providers else ['provider1', 'provider2']
    
    def fit_transform(self, df):
        """
        Apply all feature engineering steps to the input dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe from the preprocessor
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        # Make a copy to avoid modifying the original dataframe
        transformed_df = df.copy()
        
        # Apply time-based feature engineering
        transformed_df = self.add_time_features(transformed_df)
        transformed_df = self.add_cyclical_time_features(transformed_df)
        
        # Apply weather features
        transformed_df = self.add_weather_features(transformed_df)
        transformed_df = self.add_binary_weather_flags(transformed_df)
        
        # Apply wind features
        transformed_df = self.add_wind_features(transformed_df)
        
        # Apply forecast-based feature engineering
        transformed_df = self.add_forecast_features(transformed_df)
        
        # Encode categorical variables
        transformed_df = self.encode_categorical_features(transformed_df)

        # Final touches
        transformed_df = self.final_touches(transformed_df)
        
        return transformed_df
    
    def transform(self, df):
        """
        Transform a new dataset using the same feature engineering steps
        
        Args:
            df (pd.DataFrame): Input dataframe from the preprocessor
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        return self.fit_transform(df)
    
    def add_time_features(self, df):
        """
        Extract basic time features from the effectivedate column
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with added time features
        """
        # Ensure 'effectivedate' is in datetime format
        df['effectivedate'] = pd.to_datetime(df['effectivedate'])
        
        # Extract basic time features
        df['year'] = df['effectivedate'].dt.year
        df['month'] = df['effectivedate'].dt.month
        df['day'] = df['effectivedate'].dt.day
        df['day_of_week'] = df['effectivedate'].dt.dayofweek  # Monday=0, Sunday=6
        df['hour'] = df['effectivedate'].dt.hour
        
        # Add season (1: Winter, 2: Spring, 3: Summer, 4: Autumn)
        df['season'] = df['effectivedate'].dt.month % 12 // 3 + 1
        
        # Add weekend indicator
        df['is_weekend'] = df['effectivedate'].dt.dayofweek >= 5  # Saturday and Sunday
        
        # Add time of day
        df['time_of_day'] = pd.cut(df['effectivedate'].dt.hour, 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                                  right=False)
        
        # Add week of year
        df['week_of_year'] = df['effectivedate'].dt.isocalendar().week
        
        return df
    
    def encode_cyclical(self, df, column, max_val):
        """
        Encodes a cyclical feature using sine and cosine transformations.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the column.
            column (str): The name of the cyclical column to encode.
            max_val (int): The maximum value of the cyclical feature (period).
        
        Returns:
            pd.DataFrame: DataFrame with new sine and cosine columns.
        """
        df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_val)
        df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_val)
        return df
    
    def add_cyclical_time_features(self, df):
        """
        Add cyclical encodings for time features
        
        Args:
            df (pd.DataFrame): Input dataframe with time features
            
        Returns:
            pd.DataFrame: Dataframe with cyclical time features
        """
        # Encode 'hour' (0-23)
        df = self.encode_cyclical(df, 'hour', 24)
        
        # Encode 'day_of_week' (0-6)
        df = self.encode_cyclical(df, 'day_of_week', 7)
        
        # Encode 'month' (1-12)
        df = self.encode_cyclical(df, 'month', 12)
        
        # Encode 'week_of_year' (1-53)
        df = self.encode_cyclical(df, 'week_of_year', 53)
        
        # Encode 'season' (1-4)
        df = self.encode_cyclical(df, 'season', 4)
        
        return df
    
    def categorize_weather(self, code):
        """
        Categorize weather code into simplified categories
        
        Args:
            code: Weather code value
            
        Returns:
            str: Weather category
        """
        if code in {0, 1}:
            return "Clear"
        elif code in {2, 3, 51, 53, 61, 63}:
            return "Rain"
        elif code in {71, 73, 75}:
            return "Snow"
        elif code in {95, 96, 99}:
            return "Storm"
        else:
            return "Other"
    
    def add_weather_features(self, df):
        """
        Add weather-related features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with weather features
        """
        # Create weather category
        if 'weather_code_x' in df.columns:
            df['weather_category'] = df['weather_code_x'].apply(self.categorize_weather)
            df['weather_category'] = df['weather_category'].astype('category').cat.codes
            
            # Drop weather_code_y if it exists
            if 'weather_code_y' in df.columns:
                df = df.drop(columns="weather_code_y")
                
        return df
    
    def add_binary_weather_flags(self, df):
        """
        Add binary flags for specific weather conditions
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with binary weather flags
        """
        # Add precipitation flag
        if 'precipitation' in df.columns:
            df['is_raining'] = (df['precipitation'] > 0).astype(int)
            
        # Add snowfall flag
        if 'snowfall' in df.columns:
            df['is_snowing'] = (df['snowfall'] > 0).astype(int)
            
        return df
    
    def add_wind_features(self, df):
        """
        Add wind-related features including interactions and transformations
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with wind features
        """
        # Create wind component features
        if 'wind_direction_10m' in df.columns and 'wind_speed_10m' in df.columns:
            # Wind Speed × Wind Direction
            wind_dir_rad = np.radians(df['wind_direction_10m'])
            df['wind_u'] = df['wind_speed_10m'] * np.cos(wind_dir_rad)
            df['wind_v'] = df['wind_speed_10m'] * np.sin(wind_dir_rad)
            
            # Wind power features (squared and cubed values)
            df['wind_speed_squared'] = df['wind_speed_10m'] ** 2
            df['wind_speed_cubed'] = df['wind_speed_10m'] ** 3
        
        # Create wind gust component features
        if 'wind_direction_10m' in df.columns and 'wind_gusts_10m' in df.columns:
            # Gusts × Direction
            wind_gust_dir_rad = np.radians(df['wind_direction_10m'])
            df['wind_gust_u'] = df['wind_gusts_10m'] * np.cos(wind_gust_dir_rad)
            df['wind_gust_v'] = df['wind_gusts_10m'] * np.sin(wind_gust_dir_rad)
            
        return df
    
    def get_categorical_columns(self, df, threshold=20):
        """
        Identify categorical columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input dataframe
            threshold (int): Maximum number of unique values to consider a numeric column as categorical
            
        Returns:
            list: List of categorical column names
        """
        cat_cols = []
        for col in df.columns:
            # If already object or categorical, add it
            if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                cat_cols.append(col)
            # If numeric with few unique values, consider it categorical
            # Handle both numpy and pandas dtypes safely
            elif (hasattr(df[col].dtype, 'kind') and df[col].dtype.kind in 'iufc') or str(df[col].dtype).startswith(('int', 'uint', 'float')):
                if df[col].nunique() < threshold:
                    cat_cols.append(col)
        return cat_cols
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features as numeric codes
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        # Encode plant_name if it exists
        if 'plant_name' in df.columns:
            df['plant_name'] = df['plant_name'].astype('category').cat.codes

        if 'time_of_day' in df.columns:
            time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            df['time_of_day'] = pd.Categorical(df['time_of_day'], categories=time_order, ordered=True)
        
        # Import weather features list from constants
        from constants.features import WEATHER_FEATURES
        
        # Find weather columns that are categorical
        processed_cols = ['weather_category', 'time_of_day', 'plant_name', 'capacity']
        cat_cols = [col for col in self.get_categorical_columns(df) 
                   if col in WEATHER_FEATURES and col not in processed_cols and col in df.columns]
        
        # Encode each categorical column
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes
            
        return df
        
    def add_forecast_features(self, df):
        """
        Add features based on forecast data for each provider
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with forecast features
        """
        # Define time differences for the forecast columns (in minutes)
        time_differences = {
            "fc1200": 1200,
            "fc75": 75,
            "fc60": 60,
            "fc55": 55,
            "fc40": 40,
            "fc0": 0
        }
        
        # Initialize an empty DataFrame to store new features
        new_features = pd.DataFrame(index=df.index)
        
        for provider in self.providers:
            # Create ramp features (difference between the latest and the oldest forecast)
            new_features[f'{provider}_ramp'] = df[f'{provider}_fc0'] - df[f'{provider}_fc1200']
            
            # Create speed features (rate of change between consecutive forecasts)
            time_diff_items = list(time_differences.items())
            for i in range(len(time_diff_items)-1):
                key1, t1 = time_diff_items[i]
                key2, t2 = time_diff_items[i+1]
                speed_feature = f'{provider}_speed_{key1}_to_{key2}'
                new_features[speed_feature] = (df[f'{provider}_{key1}'] - df[f'{provider}_{key2}']) / (t1 - t2)
            
            # Create volatility features (range of forecast values)
            forecast_columns = [f'{provider}_{key}' for key in time_differences.keys()]
            new_features[f'{provider}_volatility'] = df[forecast_columns].max(axis=1) - df[forecast_columns].min(axis=1)
            
            # Create standard deviation feature
            new_features[f'{provider}_std_fc'] = df[forecast_columns].std(axis=1)
            
            # Create mean forecast feature
            new_features[f'{provider}_mean_fc'] = df[forecast_columns].mean(axis=1)
        
        # Combine new features with the original dataset
        return pd.concat([df, new_features], axis=1)
    
    def final_touches(self, df):
        """
        Add final touches to the dataframe
        """
        df = df.dropna()
        df = df.drop_duplicates(subset=['effectivedate', 'plant_name'])
        return df
