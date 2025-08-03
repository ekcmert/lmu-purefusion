#!/usr/bin/env python
# -*- coding: utf-8 -*-

from constants.features import MAIN_FEATURES, TIME_FEATURES, WEATHER_FEATURES, get_forecast_features


# Case dictionary with properties for each case
CASE_DIC = {
    1: {
        "features": MAIN_FEATURES + TIME_FEATURES + get_forecast_features([1, 2, 5, 6], da=False),
        "plants": ["Plant A", "Plant B", "Plant C", "Plant D", "Plant E", "Plant F", "Plant G", "Plant H", "Plant J", "Plant K"],
        "providers": ["Provider 1", "Provider 2", "Provider 5", "Provider 6"],
        "type": "Last FC"
    },
    
    2: {
        "features": MAIN_FEATURES + TIME_FEATURES + get_forecast_features([1, 2, 5, 6], da=False) + WEATHER_FEATURES,
        "plants": ["Plant A", "Plant B", "Plant C", "Plant D", "Plant E", "Plant F", "Plant G", "Plant H", "Plant J", "Plant K"],
        "providers": ["Provider 1", "Provider 2", "Provider 5", "Provider 6"],
        "type": "Last FC"
    },
    
    3: {
        "features": MAIN_FEATURES + TIME_FEATURES + get_forecast_features([1, 2, 5, 6], da=True),
        "plants": ["Plant A", "Plant B", "Plant C", "Plant D", "Plant E", "Plant F", "Plant G", "Plant H", "Plant J", "Plant K"],
        "providers": ["Provider 1", "Provider 2", "Provider 5", "Provider 6"],
        "type": "Dayahead FC"
    },
    
    4: {
        "features": MAIN_FEATURES + TIME_FEATURES + get_forecast_features([1, 2, 3, 5, 6], da=False)  + WEATHER_FEATURES,
        "plants": ["Plant C", "Plant G"],
        "providers": ["Provider 1", "Provider 2", "Provider 3", "Provider 5", "Provider 6"],
        "type": "Last FC"
    },
    
    5: {
        "features": MAIN_FEATURES + TIME_FEATURES + get_forecast_features([1, 2, 4, 5, 6], da=False) + WEATHER_FEATURES,
        "plants": ["Plant D", "Plant J", "Plant K"],
        "providers": ["Provider 1", "Provider 2", "Provider 4", "Provider 5", "Provider 6"],
        "type": "Last FC"
    },
    
    6: {
        "features": MAIN_FEATURES + TIME_FEATURES + get_forecast_features([1, 2, 3, 5, 6], da=True),
        "plants": ["Plant C", "Plant G"],
        "providers": ["Provider 1", "Provider 2", "Provider 3", "Provider 5", "Provider 6"],
        "type": "Dayahead FC"
    },
    
    7: {
        "features": MAIN_FEATURES + TIME_FEATURES + get_forecast_features([1, 2, 4, 5, 6], da=True),
        "plants": ["Plant D", "Plant J", "Plant K"],
        "providers": ["Provider 1", "Provider 2", "Provider 4", "Provider 5", "Provider 6"],
        "type": "Dayahead FC"
    }
}




