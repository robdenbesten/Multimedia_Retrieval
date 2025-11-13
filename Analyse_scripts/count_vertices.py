"""
VERTEX COUNTER
This file counts how many vertices (points) each 3D shape has.
It handles different file formats (starting with 'm' or 'D').
It scans folders and saves vertex counts to CSV files.
This helps analyze the quality of the database.
"""

import os
import re
from pathlib import Path
import csv


def count_vertices_in_file(file_path):
    """
    Count vertices in an .obj file based on its filename.
    
    Args:
        file_path (str): Path to the .obj file
        
    Returns:
        int: Number of vertices, or -1 if error occurred
    """
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if filename.startswith('m'):
                # For files starting with 'm', get vertex count from line 8
                lines = f.readlines()
                if len(lines) >= 8:
                    line8 = lines[7]  # 0-indexed, so line 8 is index 7
                    # Look for pattern like "# Vertices: 120"
                    match = re.search(r'#\s*Vertices:\s*(\d+)', line8)
                    if match:
                        return int(match.group(1))
                return -1
            
            elif filename.startswith('D'):
                # For files starting with 'D', count lines that start with 'v '
                f.seek(0)  # Reset file pointer
                vertex_count = 0
                for line in f:
                    if line.startswith('v '):
                        vertex_count += 1
                return vertex_count
            
            else:
                print(f"Warning: Unknown file format for {filename}")
                return -1
                
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return -1


def scan_folder_for_vertices(folder_path, recursive=True):
    """
    Scan a folder for .obj files and count vertices.
    
    Args:
        folder_path (str): Path to the folder to scan
        recursive (bool): Whether to scan subfolders recursively
        
    Returns:
        list: List of dictionaries with file info and vertex counts
    """
    results = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return results
    
    # Find all .obj files, excluding those ending with _rm.obj
    pattern = "**/*.obj" if recursive else "*.obj"
    obj_files = folder_path.glob(pattern)
    
    for obj_file in obj_files:
        filename = obj_file.name
        
        # Only process files ending with _rm.obj
        if not filename.endswith('_rm.obj'):
            continue
            
        vertex_count = count_vertices_in_file(str(obj_file))
        
        result = {
            'file_path': str(obj_file),
            'filename': filename,
            'folder': str(obj_file.parent),
            'vertex_count': vertex_count
        }
        results.append(result)
        
        # Print progress
        status = "✓" if vertex_count != -1 else "✗"
        print(f"{status} {filename}: {vertex_count if vertex_count != -1 else 'Error'} vertices")
    
    return results


def save_results_to_csv(results, output_file):
    """
    Save vertex count results to a CSV file as a simple list of vertex values.
    
    Args:
        results (list): List of result dictionaries
        output_file (str): Path to output CSV file
    """
    if not results:
        print("No results to save")
        return
    
    # Filter out failed results and get only vertex counts
    vertex_counts = [result['vertex_count'] for result in results if result['vertex_count'] != -1]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['vertex_count'])
        
        # Write each vertex count as a separate row
        for count in vertex_counts:
            writer.writerow([count])
    
    print(f"Results saved to {output_file} ({len(vertex_counts)} vertex counts)")


def print_summary(results):
    """
    Print a summary of the vertex count results.
    
    Args:
        results (list): List of result dictionaries
    """
    if not results:
        print("No files processed")
        return
    
    total_files = len(results)
    successful_files = len([r for r in results if r['vertex_count'] != -1])
    failed_files = total_files - successful_files
    
    valid_results = [r for r in results if r['vertex_count'] != -1]
    if valid_results:
        total_vertices = sum(r['vertex_count'] for r in valid_results)
        avg_vertices = total_vertices / len(valid_results)
        min_vertices = min(r['vertex_count'] for r in valid_results)
        max_vertices = max(r['vertex_count'] for r in valid_results)
    else:
        total_vertices = avg_vertices = min_vertices = max_vertices = 0
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total files processed: {total_files}")
    print(f"Successful: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Total vertices: {total_vertices:,}")
    if valid_results:
        print(f"Average vertices per file: {avg_vertices:.1f}")
        print(f"Min vertices: {min_vertices:,}")
        print(f"Max vertices: {max_vertices:,}")
    
    # Breakdown by main folder
    print("\nBreakdown by folder:")
    print("-" * 30)
    folder_stats = {}
    for result in valid_results:
        folder_path = result['folder']
        # Get the main folder (Normalised-objects or Excluded-objects)
        if 'Normalised-objects' in folder_path:
            main_folder = 'Normalised-objects'
        elif 'Excluded-objects' in folder_path:
            main_folder = 'Excluded-objects'
        else:
            main_folder = 'Other'
        
        if main_folder not in folder_stats:
            folder_stats[main_folder] = {'count': 0, 'vertices': 0}
        folder_stats[main_folder]['count'] += 1
        folder_stats[main_folder]['vertices'] += result['vertex_count']
    
    for folder, stats in folder_stats.items():
        avg = stats['vertices'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{folder}: {stats['count']} files, {stats['vertices']:,} vertices (avg: {avg:.1f})")


def main():
    """
    Main function to run the vertex counting script.
    """
    print("Vertex Counter for .obj files")
    print("Only processes files ending with '_rm.obj'")
    print("="*50)
    
    # Define the folder to scan
    base_path = r"c:\Users\rob_d\Desktop\GIthub\Multimedia_Retrieval"
    folder_to_scan = os.path.join(base_path, "Normalised-objects")
    
    # Ask if user wants recursive scanning
    recursive_input = input("Scan subfolders recursively? (y/n, default=y): ").strip().lower()
    recursive = recursive_input != 'n'
    
    print(f"\nScanning {'recursively' if recursive else 'non-recursively'}: {folder_to_scan}")
    print("-"*50)
    
    # Scan the folder
    all_results = []
    if os.path.exists(folder_to_scan):
        print(f"\nScanning: {os.path.basename(folder_to_scan)}")
        print("="*40)
        results = scan_folder_for_vertices(folder_to_scan, recursive)
        all_results.extend(results)
    else:
        print(f"Warning: Folder {folder_to_scan} does not exist")
    
    # Print summary
    print_summary(all_results)
    
    # Ask if user wants to save results
    if all_results:
        save_input = input("\nSave results to CSV? (y/n, default=y): ").strip().lower()
        if save_input != 'n':
            output_file = input("Enter output CSV filename (default=vertex_analysis.csv): ").strip()
            if not output_file:
                output_file = "vertex_analysis.csv"
            
            # Make sure it has .csv extension
            if not output_file.endswith('.csv'):
                output_file += '.csv'
                
            save_results_to_csv(all_results, output_file)


if __name__ == "__main__":
    main()
