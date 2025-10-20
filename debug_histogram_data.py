"""Quick test to check if histogram data exists and is correct"""
import json
import os

# Load features dictionary
with open('features_dictionary.json', 'r') as f:
    features_dict = json.load(f)

# Get first file to test
first_category = list(features_dict.keys())[0]
first_file = list(features_dict[first_category].keys())[0]
file_data = features_dict[first_category][first_file]

print(f"Testing file: {first_file} from category: {first_category}")
print(f"Available keys: {list(file_data.keys())}")

# Check histograms
histogram_keys = ['D1_hist', 'D2_hist', 'D3_hist', 'D4_hist', 'A3_hist']
histograms = []

for hist_key in histogram_keys:
    if hist_key in file_data:
        hist_data = file_data[hist_key].get('histogram_data', [])
        if hist_data:
            histograms.append(hist_data)
            print(f"✅ {hist_key}: {len(hist_data)} bins, max value: {max(hist_data):.3f}")
        else:
            print(f"❌ {hist_key}: no histogram_data")
    else:
        print(f"❌ {hist_key}: key not found")

print(f"\nTotal histograms found: {len(histograms)}")

# Check metrics
metrics = file_data.get('Metrics', {})
print(f"\nMetrics found: {len(metrics)}")
for key, value in list(metrics.items())[:3]:
    print(f"  {key}: {value}")