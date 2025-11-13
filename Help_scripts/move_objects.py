"""
OBJECT ORGANIZER
This file moves objects between folders based on their status.
It finds excluded objects and moves their original versions too.
It maintains folder structure when moving files.
This keeps the database organized and consistent.
"""

import os
import shutil
from pathlib import Path

def move_matching_objects():
    """
    Check objects in Excluded-objects folder, remove '_rm' suffix from names,
    find matching objects in ShapeDatabase_INFOMR-master, and move them to excluded folders.
    """
    # Define base paths
    excluded_base = Path("c:/Users/rob_d/Desktop/GIthub/Multimedia_Retrieval/Excluded-objects")
    database_base = Path("c:/Users/rob_d/Desktop/GIthub/Multimedia_Retrieval/ShapeDatabase_INFOMR-master/Original Database")
    
    # Check if base directories exist
    if not excluded_base.exists():
        print(f"Error: Excluded-objects folder not found at {excluded_base}")
        return
    
    if not database_base.exists():
        print(f"Error: ShapeDatabase folder not found at {database_base}")
        return
    
    moved_count = 0
    not_found_count = 0
    
    print("Starting object matching and moving process...\n")
    
    # Iterate through all category folders in Excluded-objects
    for category_folder in excluded_base.iterdir():
        if category_folder.is_dir():
            category_name = category_folder.name
            print(f"Processing category: {category_name}")
            
            # Check if corresponding category exists in database
            database_category = database_base / category_name
            if not database_category.exists():
                print(f"  Warning: Category '{category_name}' not found in database")
                continue
            
            # Process all files in the excluded category folder
            for excluded_file in category_folder.iterdir():
                if excluded_file.is_file() and excluded_file.name.endswith('_rm.obj'):
                    # Remove '_rm' suffix to get original name
                    original_name = excluded_file.name.replace('_rm.obj', '.obj')
                    
                    # Look for matching file in database
                    source_file = database_category / original_name
                    
                    if source_file.exists():
                        try:
                            # Create destination path in excluded folder
                            destination = excluded_file.parent / original_name
                            
                            # Move the file
                            shutil.move(str(source_file), str(destination))
                            print(f"  ✓ Moved {original_name} to {category_name}/")
                            moved_count += 1
                            
                        except Exception as e:
                            print(f"  ✗ Error moving {original_name}: {e}")
                    else:
                        print(f"  - {original_name} not found in database")
                        not_found_count += 1
    
    print(f"\nProcess completed!")
    print(f"Files moved: {moved_count}")
    print(f"Files not found: {not_found_count}")

def verify_matches():
    """
    Verify which objects would be matched without actually moving them.
    """
    excluded_base = Path("c:/Users/rob_d/Desktop/GIthub/Multimedia_Retrieval/Excluded-objects")
    database_base = Path("c:/Users/rob_d/Desktop/GIthub/Multimedia_Retrieval/ShapeDatabase_INFOMR-master/Original Database")
    
    print("Verification mode - checking for matches without moving files...\n")
    
    matches_found = []
    matches_not_found = []
    
    for category_folder in excluded_base.iterdir():
        if category_folder.is_dir():
            category_name = category_folder.name
            database_category = database_base / category_name
            
            if database_category.exists():
                for excluded_file in category_folder.iterdir():
                    if excluded_file.is_file() and excluded_file.name.endswith('_rm.obj'):
                        original_name = excluded_file.name.replace('_rm.obj', '.obj')
                        source_file = database_category / original_name
                        
                        if source_file.exists():
                            matches_found.append(f"{category_name}/{original_name}")
                        else:
                            matches_not_found.append(f"{category_name}/{original_name}")
    
    print("MATCHES FOUND:")
    for match in matches_found:
        print(f"  ✓ {match}")
    
    print(f"\nNOT FOUND:")
    for not_found in matches_not_found:
        print(f"  ✗ {not_found}")
    
    print(f"\nSummary: {len(matches_found)} matches found, {len(matches_not_found)} not found")

if __name__ == "__main__":
    print("Object Mover Script")
    print("==================")
    print("1. Run verification (check matches without moving)")
    print("2. Move matching objects")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        verify_matches()
    elif choice == "2":
        confirm = input("Are you sure you want to move files? (y/N): ").strip().lower()
        if confirm == 'y':
            move_matching_objects()
        else:
            print("Operation cancelled.")
    else:
        print("Invalid choice. Please run the script again.")
