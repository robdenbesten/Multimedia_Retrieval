"""Quick test to verify filename matching"""
import json
import os

# Load features dictionary
with open('features_dictionary.json', 'r') as f:
    features_dict = json.load(f)

# Test filename extraction
test_filenames = ["m1337.obj", "D00123.obj", "m0001.obj"]

for filename in test_filenames:
    base_filename = os.path.splitext(filename)[0]  # Remove .obj extension
    print(f"Testing: {filename} -> {base_filename}")
    
    found = False
    for category, files in features_dict.items():
        if base_filename in files:
            print(f"  ✅ Found {base_filename} in category {category}")
            metrics = files[base_filename].get('Metrics', {})
            print(f"     Metrics available: {len(metrics)}")
            print(f"     First 3 metrics: {list(metrics.keys())[:3]}")
            found = True
            break
    
    if not found:
        print(f"  ❌ {base_filename} not found in any category")

# Show first few entries to verify structure
print("\nFirst few entries:")
first_category = list(features_dict.keys())[0]
first_files = list(features_dict[first_category].keys())[:3]
print(f"Category: {first_category}")
print(f"First files: {first_files}")