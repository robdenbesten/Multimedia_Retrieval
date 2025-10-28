import os
import csv
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def count_vertices_in_obj(obj_file_path):
    """
    Count the number of vertices in an OBJ file.
    Vertices are lines that start with 'v ' in the OBJ format.
    
    Args:
        obj_file_path (str): Path to the OBJ file
        
    Returns:
        int: Number of vertices in the OBJ file
    """
    vertex_count = 0
    try:
        with open(obj_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                # Vertices in OBJ files start with 'v '
                if line.startswith('v '):
                    vertex_count += 1
    except Exception as e:
        print(f"Error reading file {obj_file_path}: {e}")
        return 0
    
    return vertex_count

def analyze_normalized_objects():
    """
    Analyze all .obj files in the Normalised-objects folder and create a CSV
    with category, object name, and vertex count information.
    """
    # Define the base path to the Normalised-objects folder
    base_path = Path(__file__).parent.parent / "Normalised-objects"
    
    # Verify the path exists
    if not base_path.exists():
        print(f"Error: Normalised-objects folder not found at {base_path}")
        return
    
    # List to store the results
    results = []
    
    # Get all category folders
    category_folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(category_folders)} categories to analyze...")
    
    # Process each category folder
    for category_folder in category_folders:
        category_name = category_folder.name
        print(f"Processing category: {category_name}")
        
        # Find all .obj files in the category folder
        obj_files = list(category_folder.glob("*.obj"))
        
        print(f"  Found {len(obj_files)} .obj files")
        
        # Process each .obj file
        for obj_file in obj_files:
            object_name = obj_file.stem  # Get filename without extension
            vertex_count = count_vertices_in_obj(obj_file)
            
            # Add to results
            results.append([category_name, object_name, vertex_count])
            
            print(f"    {object_name}: {vertex_count} vertices")
    
    # Create CSV file with the results
    output_file = base_path.parent / "vertex_analysis.csv"
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Category', 'Object_Name', 'Vertex_Count'])
            
            # Write all results
            writer.writerows(results)
        
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        print(f"Total objects analyzed: {len(results)}")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")

def create_histogram_from_csv():
    """
    Read the vertex_analysis.csv file and create a histogram with bins of width 5000.
    Shows categories and objects when hovering over bars.
    """
    # Path to the CSV file
    csv_path = Path(__file__).parent / "vertex_analysis.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run the vertex analysis first to generate the CSV file.")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data for {len(df)} objects")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Create bins with width 5000
    vertex_counts = df['Vertex_Count'].values
    max_vertices = int(np.ceil(vertex_counts.max() / 5000) * 5000)
    bins = np.arange(0, max_vertices + 5000, 5000)
    
    # Create histogram data
    hist_counts, bin_edges = np.histogram(vertex_counts, bins=bins)
    
    # Create a dictionary to store objects in each bin for tooltip information
    bin_data = defaultdict(list)
    
    # Assign each object to its corresponding bin
    for _, row in df.iterrows():
        vertex_count = row['Vertex_Count']
        category = row['Category']
        object_name = row['Object_Name']
        
        # Find which bin this vertex count belongs to
        bin_index = int(vertex_count // 5000)
        bin_range = f"{bin_index * 5000}-{(bin_index + 1) * 5000}"
        
        bin_data[bin_range].append({
            'category': category,
            'object': object_name,
            'vertices': vertex_count
        })
    
    # Create the histogram plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bar_centers = bin_edges[:-1] + 2500  # Center of each bin
    bars = ax.bar(bar_centers, hist_counts, width=4500, alpha=0.7, edgecolor='black')
    
    # Customize the plot
    ax.set_xlabel('Vertex Count', fontsize=12)
    ax.set_ylabel('Number of Objects', fontsize=12)
    ax.set_title('Distribution of Vertex Counts in Normalized Objects', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks and labels
    ax.set_xticks(bar_centers)
    ax.set_xticklabels([f"{int(edge)}-{int(edge + 5000)}" for edge in bin_edges[:-1]], rotation=45)
    
    # Add hover functionality
    def on_hover(event):
        if event.inaxes == ax:
            for i, bar in enumerate(bars):
                if bar.contains(event)[0]:
                    # Get bin range
                    bin_range = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
                    
                    if bin_range in bin_data:
                        objects_in_bin = bin_data[bin_range]
                        
                        # Group by category
                        category_groups = defaultdict(list)
                        for obj in objects_in_bin:
                            category_groups[obj['category']].append(obj)
                        
                        # Create tooltip text
                        tooltip_text = f"Range: {bin_range} vertices\n"
                        tooltip_text += f"Total objects: {len(objects_in_bin)}\n\n"
                        
                        for category, objects in category_groups.items():
                            tooltip_text += f"{category} ({len(objects)} objects):\n"
                            # Show up to 5 objects per category to avoid overcrowding
                            for obj in objects[:5]:
                                tooltip_text += f"  - {obj['object']} ({obj['vertices']} vertices)\n"
                            if len(objects) > 5:
                                tooltip_text += f"  ... and {len(objects) - 5} more\n"
                            tooltip_text += "\n"
                        
                        # Remove existing annotations
                        for annotation in ax.texts:
                            if hasattr(annotation, 'is_tooltip'):
                                annotation.remove()
                        
                        # Add new annotation
                        ann = ax.annotate(tooltip_text, 
                                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                        xytext=(10, 10), 
                                        textcoords='offset points',
                                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                                        fontsize=8,
                                        ha='left')
                        ann.is_tooltip = True
                        fig.canvas.draw_idle()
                        return
    
    # Connect the hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    # Display statistics
    print("\nHistogram Statistics:")
    for i, count in enumerate(hist_counts):
        if count > 0:
            bin_range = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
            print(f"  {bin_range} vertices: {count} objects")
    
    print(f"\nTotal objects: {len(df)}")
    print(f"Min vertices: {vertex_counts.min()}")
    print(f"Max vertices: {vertex_counts.max()}")
    print(f"Mean vertices: {vertex_counts.mean():.1f}")
    print(f"Median vertices: {np.median(vertex_counts):.1f}")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function - now creates histogram from existing CSV file.
    To generate the CSV file first, uncomment the line below.
    """
    # Uncomment the next line if you need to generate the CSV file first
    # analyze_normalized_objects()
    
    print("Creating histogram from vertex analysis data...")
    create_histogram_from_csv()

if __name__ == "__main__":
    main()
