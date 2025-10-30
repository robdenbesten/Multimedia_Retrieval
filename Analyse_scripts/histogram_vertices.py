import os
import sys
import pymeshlab as ml
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path to access the main folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NORMALISED_OBJECTS_FOLDER = r'Normalised-objects'

def get_vertex_count(file_path):
    """
    Get the vertex count of a mesh file using pymeshlab.
    
    Args:
        file_path (str): Path to the mesh file
        
    Returns:
        int: Number of vertices in the mesh, or -1 if error
    """
    try:
        ms = ml.MeshSet()
        ms.load_new_mesh(file_path)
        return ms.current_mesh().vertex_number()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return -1

def analyze_vertex_counts():
    """
    Analyze vertex counts of all .obj files in the Normalised-objects folder
    and create a histogram.
    """
    # Get the full path to the normalised objects folder
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(script_dir, NORMALISED_OBJECTS_FOLDER)
    
    if not os.path.exists(folder_path):
        print(f"Error: {folder_path} folder not found!")
        return
    
    print(f"Scanning {folder_path} for .obj files...")
    
    vertex_counts = []
    category_counts = defaultdict(list)
    total_files = 0
    error_files = 0
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                # Get category from directory structure
                relative_path = os.path.relpath(root, folder_path)
                category = relative_path if relative_path != '.' else 'Root'
                
                # Get vertex count
                vertex_count = get_vertex_count(file_path)
                
                if vertex_count == -1:
                    error_files += 1
                    continue
                
                vertex_counts.append(vertex_count)
                category_counts[category].append(vertex_count)
                
                print(f"Processed: {os.path.relpath(file_path, folder_path)} - {vertex_count} vertices")
    
    if not vertex_counts:
        print("No valid .obj files found!")
        return
    
    # Convert to numpy array for easier analysis
    vertex_counts = np.array(vertex_counts)
    
    # Print statistics
    print("\n" + "="*60)
    print("VERTEX COUNT ANALYSIS:")
    print(f"Total files analyzed: {len(vertex_counts)}")
    print(f"Files with errors: {error_files}")
    print(f"Min vertex count: {vertex_counts.min()}")
    print(f"Max vertex count: {vertex_counts.max()}")
    print(f"Mean vertex count: {vertex_counts.mean():.2f}")
    print(f"Median vertex count: {np.median(vertex_counts):.2f}")
    print(f"Standard deviation: {vertex_counts.std():.2f}")
    print("="*60)
    
    # Print category breakdown
    print("\nVERTEX COUNT BY CATEGORY:")
    for category, counts in category_counts.items():
        counts_array = np.array(counts)
        print(f"{category}: {len(counts)} files, mean: {counts_array.mean():.2f}, "
              f"min: {counts_array.min()}, max: {counts_array.max()}")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 1, 1)
    plt.hist(vertex_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Vertex Counts in Normalised Objects', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for target range (4000-10000)
    plt.axvline(x=4000, color='red', linestyle='--', label='Target Range (4000-10000)')
    plt.axvline(x=10000, color='red', linestyle='--')
    plt.legend()
    
    # Add statistics text
    stats_text = f'Total files: {len(vertex_counts)}\n'
    stats_text += f'Mean: {vertex_counts.mean():.0f}\n'
    stats_text += f'Median: {np.median(vertex_counts):.0f}\n'
    stats_text += f'Std: {vertex_counts.std():.0f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Zoomed histogram focusing on the target range
    plt.subplot(2, 1, 2)
    # Filter data for zoom view (extend range a bit for context)
    zoom_data = vertex_counts[(vertex_counts >= 2000) & (vertex_counts <= 15000)]
    plt.hist(zoom_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Zoomed View: Vertex Counts (2000-15000 range)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for target range
    plt.axvline(x=4000, color='red', linestyle='--', label='Target Range (4000-10000)')
    plt.axvline(x=10000, color='red', linestyle='--')
    plt.legend()
    
    # Count files in and out of range
    in_range = np.sum((vertex_counts >= 4000) & (vertex_counts <= 10000))
    out_of_range = len(vertex_counts) - in_range
    
    range_text = f'In range (4000-10000): {in_range}\n'
    range_text += f'Out of range: {out_of_range}\n'
    range_text += f'Percentage in range: {(in_range/len(vertex_counts)*100):.1f}%'
    plt.text(0.02, 0.98, range_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'vertex_count_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return vertex_counts, category_counts

if __name__ == "__main__":
    print("Vertex Count Histogram Generator")
    print("Analyzing vertex counts in Normalised-objects folder...")
    print()
    
    vertex_counts, category_counts = analyze_vertex_counts()