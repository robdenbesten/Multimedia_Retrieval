"""
VERTEX DISTRIBUTION PLOTTER
This file creates histograms showing how vertices are distributed across shapes.
It reads vertex counts from CSV files.
It creates bar charts with bins to visualize the distribution.
This helps understand if shapes are properly standardized.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_vertex_histogram():
    """
    Create a histogram of vertex counts from original_vertices.csv
    """
    # Read the CSV file
    csv_path = r"c:\Users\rob_d\Desktop\GIthub\Multimedia_Retrieval\filtered_vertices.csv"
    
    try:
        # Load the data
        df = pd.read_csv(csv_path)
        vertex_counts = df['vertex_count'].values
        
        print(f"Loaded {len(vertex_counts)} vertex counts from CSV")
        print(f"Range: {vertex_counts.min()} - {vertex_counts.max()}")
        
        # Create histogram with bin width of 1000
        min_val = int(vertex_counts.min())
        max_val = int(vertex_counts.max())
        
        # Create bins with width of 1000
        bins = np.arange(0, max_val + 1000, 1000)
        
        # Create the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(vertex_counts, bins=bins, color='blue', alpha=0.7, edgecolor='black')
        
        # Add labels and title
        plt.xlabel('Vertex Count')
        plt.ylabel('Frequency')
        plt.title('Histogram of Vertex Counts (Bin Width = 1000)')
        plt.grid(True, alpha=0.3)
        

        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Print basic range information
        print(f"\nRange: {vertex_counts.min()} - {vertex_counts.max()} vertices")
        print(f"Total files: {len(vertex_counts)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {csv_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_vertex_histogram()
