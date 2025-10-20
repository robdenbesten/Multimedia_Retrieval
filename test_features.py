"""Test script to check if the main functionality works"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Test imports
    print("Testing imports...")
    import json
    import numpy as np
    
    # Test loading features dictionary
    print("Testing features dictionary loading...")
    with open('features_dictionary.json', 'r') as f:
        features_dict = json.load(f)
    
    print(f"Features dictionary loaded successfully!")
    print(f"Number of categories: {len(features_dict)}")
    
    # Test getting first few entries
    first_category = list(features_dict.keys())[0]
    print(f"First category: {first_category}")
    
    first_file = list(features_dict[first_category].keys())[0]
    print(f"First file in category: {first_file}")
    
    file_data = features_dict[first_category][first_file]
    print(f"Available data keys: {list(file_data.keys())}")
    
    if 'Metrics' in file_data:
        metrics = file_data['Metrics']
        print(f"Number of metrics: {len(metrics)}")
        print(f"First 3 metrics: {list(metrics.keys())[:3]}")
    
    # Test histogram data
    histogram_keys = ['D1_hist', 'D2_hist', 'D3_hist', 'D4_hist', 'A3_hist']
    available_histograms = [key for key in histogram_keys if key in file_data]
    print(f"Available histograms: {available_histograms}")
    
    if available_histograms:
        first_hist = file_data[available_histograms[0]]
        if 'histogram_data' in first_hist:
            hist_data = first_hist['histogram_data']
            print(f"First histogram length: {len(hist_data)}")
    
    print("All tests passed! The features functionality should work.")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()