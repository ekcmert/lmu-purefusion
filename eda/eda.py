import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_regression
from datetime import timedelta
import os

class RenewableEnergyEDA:
    """
    A comprehensive EDA class for analyzing renewable energy forecasting data.
    This class provides various visualization and analysis methods to understand 
    the relationships between weather features, forecasts, and production.
    """
    
    def __init__(self, data_path=None, df=None, providers=None, plants=None):
        """
        Initialize the EDA class with either a path to the data or a pandas DataFrame.
        
        Args:
            data_path (str, optional): Path to the CSV file containing the data.
            df (pd.DataFrame, optional): DataFrame containing the data.
            providers (list, optional): List of forecast providers to include in the analysis.
                                      If None, all available providers will be used.
            plants (list or str, optional): Specific plant(s) to analyze. If None, all plants are included.
        """
        self.df = None
        self.plants = None
        self.providers = None
        self.selected_providers = providers  # Store the user-selected providers
        self.selected_plants = plants  # Store the user-selected plants
        self.meteorological_columns = [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
            'apparent_temperature', 'precipitation', 'rain', 'snowfall',
            'snow_depth', 'weather_code_x', 'pressure_msl', 'surface_pressure',
            'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
            'et0_fao_evapotranspiration_x', 'vapour_pressure_deficit',
            'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
            'et0_fao_evapotranspiration_y', 'temperature_2m_max',
            'temperature_2m_min', 'apparent_temperature_max',
            'apparent_temperature_min', 'daylight_duration', 'sunshine_duration',
            'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'precipitation_hours', 'wind_speed_10m_max', 'wind_gusts_10m_max',
            'wind_direction_10m_dominant', 'shortwave_radiation_sum'
        ]
        self.forecast_columns = ['fc0', 'fc1200', 'fc40', 'fc55', 'fc60', 'fc75']
        
        if data_path is not None:
            self.load_data(data_path)
        elif df is not None:
            self.df = df.copy()
            self._initial_preprocessing()
        else:
            print("No data provided. Use load_data() method to load the data.")
    
    def load_data(self, data_path):
        """
        Load data from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file.
        """
        try:
            self.df = pd.read_csv(data_path)
            print(f"Data loaded successfully with shape: {self.df.shape}")
            self._initial_preprocessing()
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def _initial_preprocessing(self):
        """
        Perform initial preprocessing on the data.
        """
        if self.df is not None:
            # Convert date column to datetime
            self.df['effectivedate'] = pd.to_datetime(self.df['effectivedate'])
            
            # Extract all unique plants and providers first
            all_plants = self.df['plant_name'].unique()
            all_providers = self.df['forecast_provider'].unique()
            
            # Apply plant filtering if specified
            if self.selected_plants is not None:
                # Convert single plant to list if necessary
                if isinstance(self.selected_plants, (str, int)):
                    plant_list = [self.selected_plants]
                else:
                    plant_list = list(self.selected_plants)
                
                # Validate that specified plants exist in the data
                valid_plants = [p for p in plant_list if p in all_plants]
                
                if len(valid_plants) == 0:
                    print(f"Warning: None of the specified plants {plant_list} found in the data. Using all plants.")
                    self.plants = all_plants
                else:
                    if len(valid_plants) < len(plant_list):
                        missing = set(plant_list) - set(valid_plants)
                        print(f"Warning: Some plants not found in the data: {missing}")
                    
                    self.plants = np.array(valid_plants)
                    
                    # Filter the dataframe to include only selected plants
                    self.df = self.df[self.df['plant_name'].isin(self.plants)]
            else:
                # Use all plants
                self.plants = all_plants
            
            # Apply provider filtering if specified
            if self.selected_providers is not None:
                # Validate that specified providers exist in the data
                valid_providers = [p for p in self.selected_providers if p in all_providers]
                
                if len(valid_providers) == 0:
                    print(f"Warning: None of the specified providers {self.selected_providers} found in the data. Using all providers.")
                    self.providers = all_providers
                else:
                    if len(valid_providers) < len(self.selected_providers):
                        missing = set(self.selected_providers) - set(valid_providers)
                        print(f"Warning: Some providers not found in the data: {missing}")
                    
                    self.providers = np.array(valid_providers)
                    
                    # Filter the dataframe to include only selected providers
                    self.df = self.df[self.df['forecast_provider'].isin(self.providers)]
            else:
                # Use all providers
                self.providers = all_providers
            
            # Add derived time features
            self._add_time_features()
            
            # Create a pivot table for provider comparison
            self._create_pivot_tables()
            
            print("Initial preprocessing completed.")
            print(f"Analyzing plants: {', '.join(map(str, self.plants))}")
            print(f"Analyzing forecast providers: {', '.join(map(str, self.providers))}")
            print(f"Filtered data shape: {self.df.shape}")
        
    def _add_time_features(self):
        """
        Add time-based features to the DataFrame.
        """
        # Extract time components
        self.df['year'] = self.df['effectivedate'].dt.year
        self.df['month'] = self.df['effectivedate'].dt.month
        self.df['day'] = self.df['effectivedate'].dt.day
        self.df['hour'] = self.df['effectivedate'].dt.hour
        self.df['day_of_week'] = self.df['effectivedate'].dt.dayofweek
        self.df['week_of_year'] = self.df['effectivedate'].dt.isocalendar().week
        
        # Create cyclical features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Create a 'season' column
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
            5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
        }
        self.df['season'] = self.df['month'].map(season_map)
        
        # Create a weekend indicator
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create time of day
        time_of_day_map = {
            0: 'Night', 1: 'Night', 2: 'Night', 3: 'Night', 4: 'Night', 5: 'Night',
            6: 'Morning', 7: 'Morning', 8: 'Morning', 9: 'Morning', 10: 'Morning', 11: 'Morning',
            12: 'Afternoon', 13: 'Afternoon', 14: 'Afternoon', 15: 'Afternoon', 16: 'Afternoon', 17: 'Afternoon',
            18: 'Evening', 19: 'Evening', 20: 'Evening', 21: 'Evening', 22: 'Evening', 23: 'Evening'
        }
        self.df['time_of_day'] = self.df['hour'].map(time_of_day_map)
        
    def _create_pivot_tables(self):
        """
        Create pivot tables for analysis.
        """
        # This will be implemented as needed
        pass
        
    def get_data_summary(self):
        """
        Get a comprehensive summary of the data.
        
        Returns:
            dict: A dictionary containing data summary information.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return None
        
        summary = {
            'shape': self.df.shape,
            'num_plants': len(self.plants),
            'num_providers': len(self.providers),
            'date_range': (self.df['effectivedate'].min(), self.df['effectivedate'].max()),
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100,
            'column_missing': self.df.isnull().sum().to_dict()
        }
        
        return summary
    
    def plot_data_availability(self, save_path=None):
        """
        Plot data availability by plant and provider.
        
        Args:
            save_path (str, optional): Path to save the plot.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Create a pivot table of data availability by plant and date
        data_by_plant = self.df.pivot_table(
            values='production',
            index='effectivedate',
            columns='plant_name',
            aggfunc='count'
        )
        
        # Plot the availability
        plt.figure(figsize=(15, 8))
        plt.title('Data Availability by Plant and Date')
        sns.heatmap(data_by_plant.notna(), cmap='viridis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()  # Close the plot instead of displaying
        
    def analyze_forecast_vs_actual(self, plant=None, provider=None, forecast_type='fc0', save_path=None):
        """
        Analyze forecast versus actual production.
        
        Args:
            plant (str, optional): Plant name to filter by.
            provider (str, optional): Provider name to filter by.
            forecast_type (str, optional): Type of forecast to analyze.
            save_path (str, optional): Path to save the plot.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
        if provider is not None:
            df_filtered = df_filtered[df_filtered['forecast_provider'] == provider]
        
        # Create a scatter plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_filtered['effectivedate'],
                y=df_filtered['production'],
                mode='lines',
                name='Actual Production'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_filtered['effectivedate'],
                y=df_filtered[forecast_type],
                mode='lines',
                name=f'{forecast_type} Forecast'
            )
        )
        
        # Calculate and display error metrics
        mae = np.mean(np.abs(df_filtered['production'] - df_filtered[forecast_type]))
        rmse = np.sqrt(np.mean((df_filtered['production'] - df_filtered[forecast_type])**2))
        
        # Add title and labels
        title = f'Forecast vs Actual Production'
        if plant is not None:
            title += f' for {plant}'
        if provider is not None:
            title += f' by {provider}'
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Production',
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text=f'MAE: {mae:.2f}<br>RMSE: {rmse:.2f}',
                    showarrow=False,
                    align='left',
                    bgcolor='rgba(255, 255, 255, 0.8)'
                )
            ]
        )
        
        if save_path:
            fig.write_html(save_path)
            # Also save as PNG
            fig.write_image(save_path.replace('.html', '.png'), width=1000, height=600)
        
    def plot_forecast_error_distribution(self, plant=None, provider=None, save_path=None):
        """
        Plot the distribution of forecast errors.
        
        Args:
            plant (str, optional): Plant name to filter by.
            provider (str, optional): Provider name to filter by.
            save_path (str, optional): Path to save the plot. Will be saved as PNG regardless of extension.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Ensure we're using PNG format to avoid issues with HTML generation
        if save_path:
            # Get base path without extension
            base_path = save_path.rsplit('.', 1)[0]
            save_path = f"{base_path}.png"
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
        if provider is not None:
            df_filtered = df_filtered[df_filtered['forecast_provider'] == provider]
        
        # Calculate errors for each forecast type
        for fc in self.forecast_columns:
            df_filtered[f'{fc}_error'] = df_filtered['production'] - df_filtered[fc]
        
        try:
            # Create a figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            # Plot histograms for each forecast type
            for i, fc in enumerate(self.forecast_columns):
                # Skip if index out of range
                if i >= len(axes):
                    break
                    
                # Create histogram
                axes[i].hist(df_filtered[f'{fc}_error'].dropna(), bins=50, alpha=0.75)
                axes[i].set_title(f'{fc} Error Distribution')
                axes[i].set_xlabel('Error')
                axes[i].set_ylabel('Frequency')
                
                # Add vertical line at zero
                axes[i].axvline(x=0, color='r', linestyle='--')
                
                # Add mean and std as text
                mean = df_filtered[f'{fc}_error'].mean()
                std = df_filtered[f'{fc}_error'].std()
                axes[i].text(0.05, 0.95, f'Mean: {mean:.2f}\nStd: {std:.2f}', 
                             transform=axes[i].transAxes, verticalalignment='top')
            
            # Add overall title
            title = 'Forecast Error Distributions'
            if plant is not None:
                title += f' for {plant}'
            if provider is not None:
                title += f' by {provider}'
                
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            # Save the figure
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                
            plt.close()
        except Exception as e:
            print(f"Error creating error distribution visualization: {str(e)}")
        
    def compare_providers(self, plant=None, forecast_type='fc0', save_path=None):
        """
        Compare forecast performance across providers.
        
        Args:
            plant (str, optional): Plant name to filter by.
            forecast_type (str, optional): Type of forecast to analyze.
            save_path (str, optional): Path to save the plot. Will be saved as PNG regardless of extension.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Ensure we're using PNG format to avoid issues with HTML generation
        if save_path:
            # Get base path without extension
            base_path = save_path.rsplit('.', 1)[0]
            save_path = f"{base_path}.png"
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
        
        # Group by provider and calculate error metrics
        providers_comparison = []
        
        for provider in self.providers:
            provider_data = df_filtered[df_filtered['forecast_provider'] == provider]
            mae = np.mean(np.abs(provider_data['production'] - provider_data[forecast_type]))
            rmse = np.sqrt(np.mean((provider_data['production'] - provider_data[forecast_type])**2))
            
            providers_comparison.append({
                'Provider': provider,
                'MAE': mae,
                'RMSE': rmse
            })
        
        providers_df = pd.DataFrame(providers_comparison)
        
        try:
            # Create a bar plot using matplotlib
            fig, ax = plt.figure(figsize=(14, 8))
            
            # Set width of bars
            barWidth = 0.35
            
            # Set positions of the bars on X axis
            r1 = np.arange(len(providers_df))
            r2 = [x + barWidth for x in r1]
            
            # Create bars
            plt.bar(r1, providers_df['MAE'], width=barWidth, label='MAE', color='indianred')
            plt.bar(r2, providers_df['RMSE'], width=barWidth, label='RMSE', color='lightsalmon')
            
            # Add labels and title
            title = f'Forecast Error Comparison by Provider ({forecast_type})'
            if plant is not None:
                title += f' for {plant}'
                
            plt.title(title)
            plt.xlabel('Provider')
            plt.ylabel('Error Value')
            plt.xticks([r + barWidth/2 for r in range(len(providers_df))], providers_df['Provider'])
            plt.legend()
            
            # Save the figure
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                
            plt.close()
        except Exception as e:
            print(f"Error creating provider comparison visualization: {str(e)}")
        
        return providers_df
    
    def analyze_weather_impact(self, plant=None, weather_feature='temperature_2m', save_path=None):
        """
        Analyze the correlation and mutual information between weather features, production, forecasts, and errors.
        Only analyzes fc1200 and fc0 forecasts, ignoring intermediate forecasts for simplicity.
        
        Args:
            plant (str, optional): Plant name to filter by.
            weather_feature (str, optional): Weather feature to analyze.
            save_path (str, optional): Path to save the plot. Will be saved as PNG regardless of extension.
            
        Returns:
            tuple: Tuple containing correlation and mutual information DataFrames
        """
        if self.df is None:
            print("No data available. Load data first.")
            return None, None
        
        # Ensure we're using PNG format to avoid issues with HTML generation
        if save_path:
            # Get base path without extension
            base_path = save_path.rsplit('.', 1)[0]
            save_path = f"{base_path}.png"
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
        
        # Check if the weather feature exists in the data
        if weather_feature not in df_filtered.columns:
            print(f"Weather feature '{weather_feature}' not found in the dataset.")
            return None, None
        
        # Check if there's enough data after filtering
        if len(df_filtered) < 10:
            print(f"Not enough data for plant={plant}, weather_feature={weather_feature}. Skipping.")
            return None, None
        
        # Create a list to store all correlation and MI results
        corr_results = []
        mi_results = []
        
        # Check for data completeness and handle missing values
        has_missing = df_filtered[weather_feature].isna().sum() > 0
        if has_missing:
            print(f"Warning: {weather_feature} has {df_filtered[weather_feature].isna().sum()} missing values. Filling with mean.")
            df_filtered[weather_feature] = df_filtered[weather_feature].fillna(df_filtered[weather_feature].mean())
        
        # Only use fc1200 and fc0 forecasts
        focus_forecasts = ['fc1200', 'fc0']
        
        # 1. Weather feature vs production
        try:
            corr_prod = df_filtered[[weather_feature, 'production']].corr().iloc[0, 1]
            corr_results.append({'Feature1': weather_feature, 'Feature2': 'production', 'Correlation': corr_prod})
        except Exception as e:
            print(f"Error calculating correlation between {weather_feature} and production: {str(e)}")
        
        # MI for weather feature vs production
        try:
            X = df_filtered[[weather_feature]].values.reshape(-1, 1)
            y = df_filtered['production'].values
            mi_prod = mutual_info_regression(X, y)[0]
            mi_results.append({'Feature1': weather_feature, 'Feature2': 'production', 'MI_Score': mi_prod})
        except Exception as e:
            print(f"Error calculating mutual information between {weather_feature} and production: {str(e)}")
        
        # 2. Weather feature vs each provider's forecasts and errors
        for provider in self.providers:
            provider_data = df_filtered[df_filtered['forecast_provider'] == provider].copy()
            
            # Skip if no data for this provider
            if len(provider_data) < 10:
                print(f"Not enough data for provider={provider}. Skipping.")
                continue
                
            # For each forecast column (only fc1200 and fc0)
            for fc in focus_forecasts:
                if fc not in provider_data.columns:
                    continue
                    
                # Skip if all values are missing
                if provider_data[fc].isna().all():
                    continue
                
                # Fill missing values with mean
                if provider_data[fc].isna().any():
                    provider_data[fc] = provider_data[fc].fillna(provider_data[fc].mean())
                
                # Correlation: Weather feature vs forecast
                try:
                    corr_fc = provider_data[[weather_feature, fc]].corr().iloc[0, 1]
                    corr_results.append({
                        'Feature1': weather_feature,
                        'Feature2': f'{provider}_{fc}',
                        'Correlation': corr_fc
                    })
                except Exception as e:
                    print(f"Error calculating correlation between {weather_feature} and {provider}_{fc}: {str(e)}")
                
                # MI: Weather feature vs forecast
                try:
                    X = provider_data[[weather_feature]].values.reshape(-1, 1)
                    y = provider_data[fc].values
                    mi_fc = mutual_info_regression(X, y)[0]
                    mi_results.append({
                        'Feature1': weather_feature,
                        'Feature2': f'{provider}_{fc}',
                        'MI_Score': mi_fc
                    })
                except Exception as e:
                    print(f"Error calculating mutual information between {weather_feature} and {provider}_{fc}: {str(e)}")
                
                # Calculate forecast error
                provider_data.loc[:, f'{fc}_error'] = np.abs(provider_data['production'] - provider_data[fc])
                
                # Correlation: Weather feature vs forecast error
                try:
                    corr_err = provider_data[[weather_feature, f'{fc}_error']].corr().iloc[0, 1]
                    corr_results.append({
                        'Feature1': weather_feature,
                        'Feature2': f'{provider}_{fc}_error',
                        'Correlation': corr_err
                    })
                except Exception as e:
                    print(f"Error calculating correlation between {weather_feature} and {provider}_{fc}_error: {str(e)}")
                
                # MI: Weather feature vs forecast error
                try:
                    X = provider_data[[weather_feature]].values.reshape(-1, 1)
                    y = provider_data[f'{fc}_error'].values
                    mi_err = mutual_info_regression(X, y)[0]
                    mi_results.append({
                        'Feature1': weather_feature,
                        'Feature2': f'{provider}_{fc}_error',
                        'MI_Score': mi_err
                    })
                except Exception as e:
                    print(f"Error calculating mutual information between {weather_feature} and {provider}_{fc}_error: {str(e)}")
        
        # Convert results to DataFrames
        if not corr_results or not mi_results:
            print(f"No valid correlation or mutual information results for {weather_feature}.")
            return None, None
            
        corr_df = pd.DataFrame(corr_results)
        mi_df = pd.DataFrame(mi_results)
        
        # Sort by absolute correlation and MI score
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)
        mi_df = mi_df.sort_values('MI_Score', ascending=False)
        
        # Create visualizations (bar charts) for correlation and MI
        if len(corr_df) > 0 and len(mi_df) > 0:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                
                # Plot correlation
                corr_df.set_index('Feature2').plot(kind='barh', y='Correlation', ax=axes[0], color='skyblue')
                axes[0].set_title(f'Correlation of {weather_feature} with Production and Forecasts')
                axes[0].set_xlabel('Correlation')
                axes[0].set_ylabel('')
                axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                
                # Plot mutual information
                mi_df.set_index('Feature2').plot(kind='barh', y='MI_Score', ax=axes[1], color='lightgreen')
                axes[1].set_title(f'Mutual Information of {weather_feature} with Production and Forecasts')
                axes[1].set_xlabel('Mutual Information Score')
                axes[1].set_ylabel('')
                
                # Add title
                title = f'Impact of {weather_feature} on Production and Forecasts'
                if plant is not None:
                    title += f' for {plant}'
                plt.suptitle(title, fontsize=16)
                
                plt.tight_layout()
                
                if save_path:
                    # Save as PNG file
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                    
                    # Save raw data as CSV
                    base_path = save_path.rsplit('.', 1)[0]  # Remove extension
                    corr_df.to_csv(f"{base_path}_correlation.csv", index=False)
                    mi_df.to_csv(f"{base_path}_mi.csv", index=False)
                
                # Close the figure to free memory
                plt.close()
            except Exception as e:
                print(f"Error creating visualization for {weather_feature}: {str(e)}")
        
        return corr_df, mi_df
    
    def correlation_analysis(self, plant=None, save_path=None):
        """
        Perform correlation analysis between features and production.
        
        Args:
            plant (str, optional): Plant name to filter by.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            pd.DataFrame: DataFrame with correlation coefficients.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
            
        # Get meteorological columns plus forecast columns
        columns_to_analyze = self.meteorological_columns + self.forecast_columns
        
        # Calculate correlations with production
        correlations = df_filtered[columns_to_analyze].corrwith(df_filtered['production'])
        correlations = correlations.sort_values(ascending=False)
        
        # Create a bar plot
        plt.figure(figsize=(14, 8))
        correlations.plot(kind='bar', color='skyblue')
        plt.title('Correlation with Production')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()  # Close the plot instead of displaying
        
        return correlations
        
    def mutual_information_analysis(self, plant=None, save_path=None):
        """
        Perform mutual information analysis between features and production.
        
        Args:
            plant (str, optional): Plant name to filter by.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            pd.DataFrame: DataFrame with mutual information scores.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
            
        # Get meteorological columns plus forecast columns
        columns_to_analyze = self.meteorological_columns + self.forecast_columns
        
        # Calculate mutual information with production
        X = df_filtered[columns_to_analyze]
        y = df_filtered['production']
        
        # Fill missing values for MI calculation
        X_filled = X.fillna(X.mean())
        
        mi_scores = mutual_info_regression(X_filled, y)
        mi_df = pd.DataFrame({'Feature': columns_to_analyze, 'MI_Score': mi_scores})
        mi_df = mi_df.sort_values('MI_Score', ascending=False)
        
        # Create a bar plot
        plt.figure(figsize=(14, 8))
        plt.barh(mi_df['Feature'], mi_df['MI_Score'], color='lightgreen')
        plt.title('Mutual Information with Production')
        plt.xlabel('Mutual Information Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()  # Close the plot instead of displaying
        
        return mi_df
    
    def combined_feature_importance(self, plant=None, save_path=None):
        """
        Combine correlation and mutual information for feature importance analysis.
        
        Args:
            plant (str, optional): Plant name to filter by.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            pd.DataFrame: DataFrame with combined feature importance scores.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
            
        # Get meteorological columns plus forecast columns
        columns_to_analyze = self.meteorological_columns + self.forecast_columns
        
        # Calculate correlations
        correlations = df_filtered[columns_to_analyze].corrwith(df_filtered['production']).abs()
        
        # Calculate mutual information
        X = df_filtered[columns_to_analyze].fillna(df_filtered[columns_to_analyze].mean())
        y = df_filtered['production']
        mi_scores = mutual_info_regression(X, y)
        
        # Combine results
        results = pd.DataFrame({
            'Feature': columns_to_analyze,
            'Abs_Correlation': correlations.values,
            'MI_Score': mi_scores
        })
        
        # Normalize scores
        results['Norm_Correlation'] = results['Abs_Correlation'] / results['Abs_Correlation'].max()
        results['Norm_MI'] = results['MI_Score'] / results['MI_Score'].max()
        
        # Combined score (average of normalized values)
        results['Combined_Score'] = (results['Norm_Correlation'] + results['Norm_MI']) / 2
        results = results.sort_values('Combined_Score', ascending=False)
        
        # Create plot
        fig = plt.figure(figsize=(15, 10))
        
        # Correlation plot
        plt.subplot(1, 2, 1)
        plt.barh(results['Feature'], results['Abs_Correlation'], color='skyblue')
        plt.xlabel('Absolute Correlation with Production')
        plt.title('Correlation Analysis')
        plt.gca().invert_yaxis()
        
        # Mutual Information plot
        plt.subplot(1, 2, 2)
        plt.barh(results['Feature'], results['MI_Score'], color='lightgreen')
        plt.xlabel('Mutual Information with Production')
        plt.title('Mutual Information Analysis')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()  # Close the plot instead of displaying
        
        return results
        
    def analyze_temporal_patterns(self, plant=None, time_unit='hour', save_path=None):
        """
        Analyze temporal patterns in production and forecast errors.
        
        Args:
            plant (str, optional): Plant name to filter by.
            time_unit (str, optional): Time unit for analysis ('hour', 'day_of_week', 'month', 'season').
            save_path (str, optional): Path to save the plot. Will be saved as PNG regardless of extension.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Ensure we're using PNG format to avoid issues with HTML generation
        if save_path:
            # Get base path without extension
            base_path = save_path.rsplit('.', 1)[0]
            save_path = f"{base_path}.png"
        
        # Filter the data if needed
        df_filtered = self.df.copy()
        if plant is not None:
            df_filtered = df_filtered[df_filtered['plant_name'] == plant]
        
        # Calculate forecast errors
        for fc in self.forecast_columns:
            df_filtered[f'{fc}_error'] = np.abs(df_filtered['production'] - df_filtered[fc])
        
        # Group by the specified time unit
        if time_unit == 'hour':
            grouped = df_filtered.groupby('hour')
            x_label = 'Hour of Day'
            title_unit = 'Hourly'
        elif time_unit == 'day_of_week':
            grouped = df_filtered.groupby('day_of_week')
            x_label = 'Day of Week (0=Monday, 6=Sunday)'
            title_unit = 'Daily'
        elif time_unit == 'month':
            grouped = df_filtered.groupby('month')
            x_label = 'Month'
            title_unit = 'Monthly'
        elif time_unit == 'season':
            grouped = df_filtered.groupby('season')
            x_label = 'Season'
            title_unit = 'Seasonal'
        else:
            print(f"Invalid time unit: {time_unit}. Use 'hour', 'day_of_week', 'month', or 'season'.")
            return
        
        # Calculate average production and errors
        avg_production = grouped['production'].mean()
        avg_errors = {fc: grouped[f'{fc}_error'].mean() for fc in self.forecast_columns}
        
        try:
            # Create a matplotlib figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(14, 12))
            
            # Plot average production
            avg_production.plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title(f'{title_unit} Average Production')
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel('Average Production')
            
            # Plot average errors
            for fc, errors in avg_errors.items():
                errors.plot(kind='bar', ax=axes[1], alpha=0.7, label=f'{fc} Error')
            
            axes[1].set_title(f'{title_unit} Average Forecast Errors')
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel('Average Absolute Error')
            axes[1].legend()
            
            # Add overall title
            title = f'{title_unit} Patterns in Production and Forecast Errors'
            if plant is not None:
                title += f' for {plant}'
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
            plt.close()
        except Exception as e:
            print(f"Error creating temporal patterns visualization: {str(e)}")

    def create_wind_features(self):
        """
        Create additional wind-related features.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Convert wind direction to radians
        wind_dir_rad = np.deg2rad(self.df['wind_direction_10m'])
        
        # Calculate wind vector components
        self.df['wind_u'] = -self.df['wind_speed_10m'] * np.sin(wind_dir_rad)
        self.df['wind_v'] = -self.df['wind_speed_10m'] * np.cos(wind_dir_rad)
        
        # Same for gusts
        wind_gust_rad = np.deg2rad(self.df['wind_direction_10m'])  # Assuming same direction for gusts
        self.df['wind_gust_u'] = -self.df['wind_gusts_10m'] * np.sin(wind_gust_rad)
        self.df['wind_gust_v'] = -self.df['wind_gusts_10m'] * np.cos(wind_gust_rad)
        
        # Quadratic and cubic terms for wind speed (important for power conversion)
        self.df['wind_speed_squared'] = self.df['wind_speed_10m'] ** 2
        self.df['wind_speed_cubed'] = self.df['wind_speed_10m'] ** 3
        
        # Wind stability (variation between gust and average)
        self.df['wind_stability'] = self.df['wind_gusts_10m'] - self.df['wind_speed_10m']
        
        print("Wind features created successfully.")
    
    def create_weather_categories(self):
        """
        Create categorical features from weather codes.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Define weather categories based on WMO codes
        # Reference: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
        weather_categories = {
            0: 'Clear',
            1: 'Mainly Clear', 
            2: 'Partly Cloudy', 
            3: 'Overcast',
            45: 'Fog', 
            48: 'Depositing Rime Fog',
            51: 'Light Drizzle', 
            53: 'Moderate Drizzle', 
            55: 'Dense Drizzle',
            56: 'Light Freezing Drizzle', 
            57: 'Dense Freezing Drizzle',
            61: 'Slight Rain', 
            63: 'Moderate Rain', 
            65: 'Heavy Rain',
            66: 'Light Freezing Rain', 
            67: 'Heavy Freezing Rain',
            71: 'Slight Snow Fall', 
            73: 'Moderate Snow Fall', 
            75: 'Heavy Snow Fall',
            77: 'Snow Grains',
            80: 'Slight Rain Showers', 
            81: 'Moderate Rain Showers', 
            82: 'Violent Rain Showers',
            85: 'Slight Snow Showers', 
            86: 'Heavy Snow Showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with Slight Hail', 
            99: 'Thunderstorm with Heavy Hail'
        }
        
        # Map weather codes to categories
        self.df['weather_category_x'] = self.df['weather_code_x'].map(weather_categories)
        
        # Create binary precipitation indicators
        self.df['is_raining'] = ((self.df['rain'] > 0) | 
                                (self.df['precipitation'] > 0) | 
                                (self.df['weather_code_x'].isin([51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]))).astype(int)
        
        self.df['is_snowing'] = ((self.df['snowfall'] > 0) | 
                                (self.df['snow_depth'] > 0) | 
                                (self.df['weather_code_x'].isin([71, 73, 75, 77, 85, 86]))).astype(int)
        
        print("Weather categories created successfully.")
    
    def create_provider_comparison_features(self):
        """
        Create features that compare forecasts between providers.
        """
        if self.df is None or len(self.providers) < 2:
            print("No data available or fewer than 2 providers. Load data first.")
            return
        
        provider_pivot = self.df.pivot_table(
            index=['effectivedate', 'plant_name'],
            columns='forecast_provider',
            values=self.forecast_columns
        )
        
        # Flatten the column multi-index
        provider_pivot.columns = ['_'.join(col).strip() for col in provider_pivot.columns.values]
        
        # Reset the index to merge back with the original dataframe
        provider_pivot = provider_pivot.reset_index()
        
        # Merge with the original dataframe
        self.df = pd.merge(
            self.df, 
            provider_pivot,
            on=['effectivedate', 'plant_name'],
            how='left'
        )
        
        # Create difference features between providers
        providers_list = list(self.providers)
        for i in range(len(providers_list)):
            for j in range(i+1, len(providers_list)):
                p1 = providers_list[i]
                p2 = providers_list[j]
                
                for fc in self.forecast_columns:
                    diff_col = f'{p1}_{p2}_{fc}_diff'
                    self.df[diff_col] = self.df[f'{fc}_{p1}'] - self.df[f'{fc}_{p2}']
        
        print("Provider comparison features created successfully.")
    
    def create_forecast_volatility_features(self):
        """
        Create features that measure the volatility and ramp rate of forecasts.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        for provider in self.providers:
            provider_data = self.df[self.df['forecast_provider'] == provider].copy()
            
            # Sort by date for each plant
            provider_data = provider_data.sort_values(['plant_name', 'effectivedate'])
            
            # Calculate ramps and speeds between forecasts
            provider_data[f'{provider}_ramp'] = provider_data['fc0'] - provider_data['fc1200']
            provider_data[f'{provider}_speed_fc1200_to_fc75'] = (provider_data['fc1200'] - provider_data['fc75']) / (1200 - 75)
            provider_data[f'{provider}_speed_fc75_to_fc60'] = (provider_data['fc75'] - provider_data['fc60']) / (75 - 60)
            provider_data[f'{provider}_speed_fc60_to_fc55'] = (provider_data['fc60'] - provider_data['fc55']) / (60 - 55)
            provider_data[f'{provider}_speed_fc55_to_fc40'] = (provider_data['fc55'] - provider_data['fc40']) / (55 - 40)
            provider_data[f'{provider}_speed_fc40_to_fc0'] = (provider_data['fc40'] - provider_data['fc0']) / 40
            
            # Calculate forecast volatility statistics
            forecast_cols = [col for col in provider_data.columns if col in self.forecast_columns]
            provider_data[f'{provider}_volatility'] = provider_data[forecast_cols].std(axis=1)
            provider_data[f'{provider}_std_fc'] = provider_data[forecast_cols].std(axis=1)
            provider_data[f'{provider}_mean_fc'] = provider_data[forecast_cols].mean(axis=1)
            
            # Merge back with the main dataframe
            self.df = pd.merge(
                self.df,
                provider_data[['plant_name', 'effectivedate'] + 
                             [col for col in provider_data.columns if provider in col and col not in self.df.columns]],
                on=['plant_name', 'effectivedate'],
                how='left'
            )
        
        print("Forecast volatility features created successfully.")
    
    def save_processed_data(self, output_path):
        """
        Save the processed data to a CSV file.
        
        Args:
            output_path (str): Path to save the processed data.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    
    def run_complete_eda(self, output_dir='eda_results', aggregate_analysis=True, 
                     sections=None):
        """
        Run a complete EDA pipeline and save all results.
        
        Args:
            output_dir (str, optional): Directory to save the EDA results.
            aggregate_analysis (bool, optional): Whether to perform analysis on the entire dataset without plant grouping.
            sections (list, optional): Specific sections of the EDA to run. Available options:
                - 'data_summary': Basic data summary statistics
                - 'data_availability': Data availability visualization
                - 'weather_impact': Weather impact analysis for key features
                - 'temporal_patterns': Analysis of temporal patterns (hourly, daily, monthly, seasonal)
                - 'provider_comparison': Comparison of forecast providers
                - 'error_distribution': Analysis of forecast error distributions
                - 'feature_importance': Feature importance analysis (correlation and mutual information)
                If None, all sections will be run.
        """
        if self.df is None:
            print("No data available. Load data first.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If no sections specified, run all
        if sections is None:
            sections = [
                'data_summary', 
                'data_availability',
                'weather_impact',
                'temporal_patterns',
                'provider_comparison',
                'error_distribution',
                'feature_importance'
            ]
        
        print(f"Running EDA pipeline with sections: {', '.join(sections)}")
        
        # 1. Data summary
        if 'data_summary' in sections:
            print("Running data summary analysis...")
            summary = self.get_data_summary()
            summary_df = pd.DataFrame({
                'Metric': list(summary.keys()),
                'Value': list(summary.values())
            })
            summary_df.to_csv(f"{output_dir}/data_summary.csv", index=False)
        
        # 2. Data availability plot
        if 'data_availability' in sections:
            print("Running data availability analysis...")
            self.plot_data_availability(save_path=f"{output_dir}/data_availability.png")
        
        # ==================== AGGREGATE ANALYSIS ==================== 
        if aggregate_analysis:
            print("Performing aggregate analysis on entire dataset...")
            
            # 3. Feature importance analysis
            if 'feature_importance' in sections:
                print("Running feature importance analysis for aggregate data...")
                # Correlation and mutual information analysis
                correlations = self.correlation_analysis(
                    plant=None, 
                    save_path=f"{output_dir}/correlations_aggregate.png"
                )
                
                if correlations is not None:
                    correlations.to_csv(
                        f"{output_dir}/correlations_aggregate.csv", 
                        index=True
                    )
                
                mi_scores = self.mutual_information_analysis(
                    plant=None,
                    save_path=f"{output_dir}/mutual_info_aggregate.png"
                )
                
                if mi_scores is not None:
                    mi_scores.to_csv(
                        f"{output_dir}/mutual_info_aggregate.csv", 
                        index=False
                    )
                
                # Combined feature importance
                feature_importance = self.combined_feature_importance(
                    plant=None,
                    save_path=f"{output_dir}/feature_importance_aggregate.png"
                )
                
                if feature_importance is not None:
                    feature_importance.to_csv(
                        f"{output_dir}/feature_importance_aggregate.csv",
                        index=False
                    )
            
            # 4. Temporal patterns analysis
            if 'temporal_patterns' in sections:
                print("Running temporal patterns analysis for aggregate data...")
                for time_unit in ['hour', 'day_of_week', 'month', 'season']:
                    self.analyze_temporal_patterns(
                        plant=None,
                        time_unit=time_unit,
                        save_path=f"{output_dir}/temporal_{time_unit}_aggregate.png"
                    )
            
            # 5. Weather impact analysis
            if 'weather_impact' in sections:
                print("Running weather impact analysis for aggregate data...")
                key_weather_features = [
                    'temperature_2m', 'wind_speed_10m', 'cloud_cover', 
                    'surface_pressure', 'shortwave_radiation_sum', 'daylight_duration'
                ]
                
                for i, feature in enumerate(key_weather_features):
                    print(f"Analyzing weather feature {i+1}/{len(key_weather_features)} for aggregate data: {feature}")
                    try:
                        self.analyze_weather_impact(
                            plant=None,
                            weather_feature=feature,
                            save_path=f"{output_dir}/weather_impact_{feature}_aggregate.png"
                        )
                        print(f"✓ Completed analysis for {feature}")
                    except Exception as e:
                        print(f"Error analyzing {feature} for aggregate data: {str(e)}")
                        print(f"Skipping {feature} and continuing with next feature...")
                        continue
            
            # 6. Provider comparison
            if 'provider_comparison' in sections:
                print("Running provider comparison for aggregate data...")
                provider_comparison = self.compare_providers(
                    plant=None,
                    save_path=f"{output_dir}/provider_comparison_aggregate.png"
                )
                
                if provider_comparison is not None:
                    provider_comparison.to_csv(
                        f"{output_dir}/provider_comparison_aggregate.csv",
                        index=False
                    )
            
            # 7. Forecast error distribution
            if 'error_distribution' in sections:
                print("Running error distribution analysis for aggregate data...")
                for provider in self.providers:
                    self.plot_forecast_error_distribution(
                        plant=None,
                        provider=provider,
                        save_path=f"{output_dir}/error_distribution_{provider}_aggregate.png"
                    )
        
        # ==================== PLANT-LEVEL ANALYSIS ====================
        # Analyze each plant individually
        if not aggregate_analysis:
            print(f"Performing plant-level analysis for: {', '.join(map(str, self.plants))}")
            
            # For each plant
            for plant in self.plants:
                print(f"Analyzing plant: {plant}")
                plant_dir = f"{output_dir}/plant_{plant}"
                os.makedirs(plant_dir, exist_ok=True)
                
                # 3. Feature importance analysis
                if 'feature_importance' in sections:
                    # Correlation and mutual information analysis
                    correlations = self.correlation_analysis(
                        plant=plant, 
                        save_path=f"{plant_dir}/correlations.png"
                    )
                    
                    if correlations is not None:
                        correlations.to_csv(
                            f"{plant_dir}/correlations.csv", 
                            index=True
                        )
                    
                    mi_scores = self.mutual_information_analysis(
                        plant=plant,
                        save_path=f"{plant_dir}/mutual_info.png"
                    )
                    
                    if mi_scores is not None:
                        mi_scores.to_csv(
                            f"{plant_dir}/mutual_info.csv", 
                            index=False
                        )
                    
                    # Combined feature importance
                    feature_importance = self.combined_feature_importance(
                        plant=plant,
                        save_path=f"{plant_dir}/feature_importance.png"
                    )
                    
                    if feature_importance is not None:
                        feature_importance.to_csv(
                            f"{plant_dir}/feature_importance.csv",
                            index=False
                        )
                
                # 4. Temporal patterns analysis
                if 'temporal_patterns' in sections:
                    for time_unit in ['hour', 'day_of_week', 'month', 'season']:
                        self.analyze_temporal_patterns(
                            plant=plant,
                            time_unit=time_unit,
                            save_path=f"{plant_dir}/temporal_{time_unit}.png"
                        )
                
                # 5. Weather impact analysis
                if 'weather_impact' in sections:
                    key_weather_features = [
                        'temperature_2m', 'wind_speed_10m', 'cloud_cover', 
                        'surface_pressure', 'shortwave_radiation_sum', 'daylight_duration'
                    ]
                    
                    for i, feature in enumerate(key_weather_features):
                        print(f"Analyzing weather feature {i+1}/{len(key_weather_features)} for plant {plant}: {feature}")
                        try:
                            self.analyze_weather_impact(
                                plant=plant,
                                weather_feature=feature,
                                save_path=f"{plant_dir}/weather_impact_{feature}.png"
                            )
                            print(f"✓ Completed analysis for {plant} - {feature}")
                        except Exception as e:
                            print(f"Error analyzing {feature} for plant {plant}: {str(e)}")
                            print(f"Skipping {feature} for this plant and continuing...")
                            continue
                
                # 6. Provider comparison
                if 'provider_comparison' in sections:
                    provider_comparison = self.compare_providers(
                        plant=plant,
                        save_path=f"{plant_dir}/provider_comparison.png"
                    )
                    
                    if provider_comparison is not None:
                        provider_comparison.to_csv(
                            f"{plant_dir}/provider_comparison.csv",
                            index=False
                        )
                
                # 7. Forecast error distribution
                if 'error_distribution' in sections:
                    for provider in self.providers:
                        self.plot_forecast_error_distribution(
                            plant=plant,
                            provider=provider,
                            save_path=f"{plant_dir}/error_distribution_{provider}.png"
                        )
        
        print(f"EDA pipeline finished. Results saved to {output_dir}")


# Main execution
if __name__ == "__main__":
    import os
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory
    project_root = os.path.dirname(current_dir)
    
    # Initialize the EDA class with data and filtering options
    data_path = os.path.join(project_root, 'data', 'fc_data.csv')
    
    # Example: Initialize with specific plants and providers
    # eda = RenewableEnergyEDA(
    #     data_path=data_path,
    #     plants=['Plant A', 'Plant B'],
    #     providers=['Provider X', 'Provider Y']
    # )
    
    # Initialize with all plants and providers
    eda = RenewableEnergyEDA(data_path=data_path)
    
    # Create additional features
    eda.create_wind_features()
    eda.create_weather_categories()
    eda.create_provider_comparison_features()
    eda.create_forecast_volatility_features()
    
    # Example: Run only specific sections of the EDA
    output_dir = os.path.join(project_root, 'results', 'eda_results')
    
    # Run only data summary and availability analysis
    # eda.run_complete_eda(
    #     output_dir=output_dir,
    #     sections=['data_summary', 'data_availability']
    # )
    
    # Run only weather impact analysis
    # eda.run_complete_eda(
    #     output_dir=output_dir,
    #     sections=['weather_impact'],
    #     aggregate_analysis=False
    # )
    
    # Run the complete EDA pipeline (all sections)
    eda.run_complete_eda(output_dir=output_dir, aggregate_analysis=True, sections=["weather_impact"])
    
    # Save the processed data
    processed_data_path = os.path.join(project_root, 'data', 'processed_data.csv')
    eda.save_processed_data(processed_data_path)
