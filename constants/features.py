#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Main features list
MAIN_FEATURES = [
    'effectivedate',
    'plant_name',
    'production',
    'capacity',
]

# Time features list
TIME_FEATURES = [
    'year', 'month', 'day', 'day_of_week', 'hour', 'season', 'is_weekend',
    'time_of_day', 'week_of_year', 'hour_sin', 'hour_cos', 'day_of_week_sin',
    'day_of_week_cos', 'month_sin', 'month_cos', 'week_of_year_sin',
    'week_of_year_cos', 'season_sin', 'season_cos',
]

# Weather features list
WEATHER_FEATURES = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
    'precipitation', 'rain', 'showers', 'snowfall', 'snow_depth', 'weather_code_x',
    'pressure_msl', 'surface_pressure', 'cloud_cover', 'cloud_cover_low',
    'cloud_cover_mid', 'cloud_cover_high', 'et0_fao_evapotranspiration_x',
    'vapour_pressure_deficit', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'et0_fao_evapotranspiration_y', 'temperature_2m_max', 'temperature_2m_min',
    'apparent_temperature_max', 'apparent_temperature_min', 'daylight_duration',
    'sunshine_duration', 'precipitation_sum', 'rain_sum', 'showers_sum', 'snowfall_sum',
    'precipitation_hours', 'wind_speed_10m_max', 'wind_gusts_10m_max',
    'wind_direction_10m_dominant', 'shortwave_radiation_sum', 'weather_category',
    'is_raining', 'is_snowing', 'wind_u', 'wind_v', 'wind_speed_squared',
    'wind_speed_cubed', 'wind_gust_u', 'wind_gust_v',
]

def get_forecast_features(provider_numbers=None, da=False):
    """
    Generate forecast feature names based on provider numbers.
    
    Args:
        provider_numbers (list, optional): List of provider numbers. Defaults to [1, 2].
    
    Returns:
        list: List of forecast feature names
    """
    if provider_numbers is None:
        provider_numbers = [1, 2]
    
    forecast_features = []
    
    if not da:

        # Basic forecast features
        forecast_horizons = ['fc0', 'fc1200', 'fc40', 'fc55', 'fc60', 'fc75']
        
        for provider in provider_numbers:
            # Add basic forecasts
            for horizon in forecast_horizons:
                forecast_features.append(f'provider{provider}_{horizon}')
            
            # Add derived forecast features
            forecast_features.append(f'provider{provider}_ramp')
            forecast_features.append(f'provider{provider}_speed_fc1200_to_fc75')
            forecast_features.append(f'provider{provider}_speed_fc75_to_fc60')
            forecast_features.append(f'provider{provider}_speed_fc60_to_fc55')
            forecast_features.append(f'provider{provider}_speed_fc55_to_fc40')
            forecast_features.append(f'provider{provider}_speed_fc40_to_fc0')
            forecast_features.append(f'provider{provider}_volatility')
            forecast_features.append(f'provider{provider}_std_fc')
            forecast_features.append(f'provider{provider}_mean_fc')
    
    else:
        for provider in provider_numbers:
            forecast_features.append(f'provider{provider}_fc1200')

        
    return forecast_features

# Default forecast features (for providers 1 and 2)
FORECAST_FEATURES = get_forecast_features()

# All features combined
def get_all_features(provider_numbers=None):
    """
    Get all features combined.
    
    Args:
        provider_numbers (list, optional): List of provider numbers for forecast features.
    
    Returns:
        list: All features combined
    """
    forecast_features = get_forecast_features(provider_numbers)
    return MAIN_FEATURES + TIME_FEATURES + WEATHER_FEATURES + forecast_features
