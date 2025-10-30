import os
import shutil
import pymeshlab as ml

NORMALISED_OBJECTS_FOLDER = r'Normalised-objects'
EXCLUDED_FOLDER = r'Excluded-objects'
MIN_VERTICES = 4000
MAX_VERTICES = 10000

def get_vertex_count(file_path):
    try:
        ms = ml.MeshSet()
        ms.load_new_mesh(file_path)
        return ms.current_mesh().vertex_number()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return -1

def move_file_with_structure(source_path, source_root, destination_root):
    # Get relative path from source root
    relative_path = os.path.relpath(source_path, source_root)
    
    # Create destination path
    destination_path = os.path.join(destination_root, relative_path)
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # Move the file
    shutil.move(source_path, destination_path)
    print(f"Moved: {relative_path}")

def filter_objects_by_vertex_count():
    if not os.path.exists(NORMALISED_OBJECTS_FOLDER):
        print(f"Error: {NORMALISED_OBJECTS_FOLDER} folder not found!")
        return
    
    # Create excluded folder if it doesn't exist
    os.makedirs(EXCLUDED_FOLDER, exist_ok=True)
    
    total_files = 0
    excluded_files = 0
    kept_files = 0
    error_files = 0
    
    print(f"Scanning {NORMALISED_OBJECTS_FOLDER} for .obj files...")
    print(f"Vertex count criteria: {MIN_VERTICES} - {MAX_VERTICES}")
    print("-" * 60)
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(NORMALISED_OBJECTS_FOLDER):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                # Get vertex count
                vertex_count = get_vertex_count(file_path)
                
                if vertex_count == -1:
                    error_files += 1
                    continue
                
                print(f"Processing: {os.path.relpath(file_path, NORMALISED_OBJECTS_FOLDER)}")
                print(f"  Vertex count: {vertex_count}")
                
                # Check if vertex count is within range
                if MIN_VERTICES <= vertex_count <= MAX_VERTICES:
                    print(f"  ✓ Keeping (within range)")
                    kept_files += 1
                else:
                    print(f"  ✗ Moving to excluded (outside range)")
                    move_file_with_structure(file_path, NORMALISED_OBJECTS_FOLDER, EXCLUDED_FOLDER)
                    excluded_files += 1
                
                print()
    
    # Print summary
    print("=" * 60)
    print("FILTERING SUMMARY:")
    print(f"Total files processed: {total_files}")
    print(f"Files kept (4000-10000 vertices): {kept_files}")
    print(f"Files excluded (outside range): {excluded_files}")
    print(f"Files with errors: {error_files}")
    print("=" * 60)

if __name__ == "__main__":
    print("Mesh Vertex Count Filter")
    print("This script filters objects based on vertex count (4000-10000 range)")
    print()
    
    # Ask for confirmation
    response = input(f"Are you sure you want to move objects outside the vertex range to '{EXCLUDED_FOLDER}'? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        filter_objects_by_vertex_count()
        print("Filtering complete!")
    else:
        print("Operation cancelled.")