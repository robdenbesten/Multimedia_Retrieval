"""
Script to convert all feature text files from the Features directory
into a single hierarchical dictionary structure and save it as a JSON file.

Structure: {category: {object_name: {feature_type: values}}}
"""

import os
import json
from typing import Dict, Any, List


def parse_feature_file(file_path: str) -> Dict[str, Any]:
    """
    Parse a single feature text file and return a dictionary with features.
    
    Args:
        file_path: Path to the feature text file
        
    Returns:
        Dictionary containing parsed features
    """
    features = {}
    current_section = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a section header (ends with ':')
                if line.endswith(':'):
                    current_section = line[:-1]  # Remove the ':'
                    features[current_section] = {}
                elif current_section and ':' in line:
                    # This is a key-value pair within a section
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert to appropriate data type
                    if ',' in value:
                        # This is likely a list/array (histogram data)
                        try:
                            # Convert comma-separated values to list of floats
                            features[current_section][key] = [float(x.strip()) for x in value.split(',') if x.strip()]
                        except ValueError:
                            # If conversion fails, keep as string
                            features[current_section][key] = value
                    else:
                        # Single value
                        try:
                            # Try to convert to float if possible
                            features[current_section][key] = float(value)
                        except ValueError:
                            # Keep as string if not a number
                            features[current_section][key] = value
                elif current_section:
                    # This might be continuation data (like histogram values on separate lines)
                    if line and ',' in line:
                        try:
                            # Assume this is histogram data
                            hist_values = [float(x.strip()) for x in line.split(',') if x.strip()]
                            if 'histogram_data' not in features[current_section]:
                                features[current_section]['histogram_data'] = []
                            features[current_section]['histogram_data'].extend(hist_values)
                        except ValueError:
                            pass  # Ignore if can't parse
    
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return {}
    
    return features


def extract_object_name(filename: str) -> str:
    """
    Extract object name from filename by removing '_copy.txt' suffix.
    
    Args:
        filename: The filename (e.g., 'm1337_copy.txt')
        
    Returns:
        Object name (e.g., 'm1337')
    """
    if filename.endswith('_copy.txt'):
        return filename[:-9]  # Remove '_copy.txt'
    elif filename.endswith('.txt'):
        return filename[:-4]  # Remove '.txt'
    else:
        return filename


def build_features_dictionary(features_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build a hierarchical dictionary from all feature files.
    
    Args:
        features_dir: Path to the Features directory
        
    Returns:
        Hierarchical dictionary: {category: {object_name: features}}
    """
    features_dict = {}
    
    if not os.path.exists(features_dir):
        print(f"Features directory not found: {features_dir}")
        return features_dict
    
    # Iterate through all category directories
    for category in os.listdir(features_dir):
        category_path = os.path.join(features_dir, category)
        
        if not os.path.isdir(category_path):
            continue
            
        print(f"Processing category: {category}")
        features_dict[category] = {}
        
        # Iterate through all feature files in the category
        feature_files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
        
        for filename in feature_files:
            file_path = os.path.join(category_path, filename)
            object_name = extract_object_name(filename)
            
            # Parse the feature file
            features = parse_feature_file(file_path)
            
            if features:  # Only add if parsing was successful
                features_dict[category][object_name] = features
                
        print(f"  - Processed {len(feature_files)} objects in {category}")
    
    return features_dict


def save_features_dictionary(features_dict: Dict[str, Dict[str, Dict[str, Any]]], output_path: str) -> None:
    """
    Save the features dictionary to a JSON file.
    
    Args:
        features_dict: The hierarchical features dictionary
        output_path: Path where to save the JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features_dict, f, indent=2, ensure_ascii=False)
        print(f"Features dictionary saved to: {output_path}")
    except Exception as e:
        print(f"Error saving features dictionary: {e}")


def print_summary(features_dict: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
    Print a summary of the features dictionary.
    
    Args:
        features_dict: The hierarchical features dictionary
    """
    total_objects = 0
    print("\n" + "="*50)
    print("FEATURES DICTIONARY SUMMARY")
    print("="*50)
    
    for category, objects in features_dict.items():
        object_count = len(objects)
        total_objects += object_count
        print(f"{category}: {object_count} objects")
        
        # Show sample feature types from first object (if any)
        if objects:
            sample_object = next(iter(objects.values()))
            feature_types = list(sample_object.keys())
            print(f"  Feature types: {feature_types}")
    
    print(f"\nTotal categories: {len(features_dict)}")
    print(f"Total objects: {total_objects}")
    print("="*50)


def main():
    """Main function to convert Features directory to dictionary."""
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(base_dir, 'Features')
    output_file = os.path.join(base_dir, 'features_dictionary.json')
    
    print("Converting Features directory to dictionary...")
    print(f"Features directory: {features_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Build the features dictionary
    features_dict = build_features_dictionary(features_dir)
    
    if not features_dict:
        print("No features found or error occurred.")
        return
    
    # Save to JSON file
    save_features_dictionary(features_dict, output_file)
    
    # Print summary
    print_summary(features_dict)
    
    # Show sample data structure
    print("\nSample data structure:")
    if features_dict:
        sample_category = next(iter(features_dict.keys()))
        sample_objects = features_dict[sample_category]
        if sample_objects:
            sample_object_name = next(iter(sample_objects.keys()))
            sample_features = sample_objects[sample_object_name]
            print(f"Category: {sample_category}")
            print(f"Object: {sample_object_name}")
            print(f"Features: {json.dumps(sample_features, indent=2)[:500]}...")


if __name__ == "__main__":
    main()