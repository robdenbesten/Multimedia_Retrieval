import os
from pathlib import Path
from collections import defaultdict

def get_objects_from_folder(base_path, folder_name):
    """
    Get all .obj files from a folder structure, organized by category.
    Returns a dictionary where keys are category names and values are sets of normalized object names.
    Normalizes names by removing '_rm' suffix to treat remeshed and original objects as the same.
    """
    folder_path = Path(base_path) / folder_name
    objects_by_category = defaultdict(set)
    
    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return objects_by_category
    
    for category_folder in folder_path.iterdir():
        if category_folder.is_dir():
            category_name = category_folder.name
            
            for obj_file in category_folder.iterdir():
                if obj_file.is_file() and obj_file.name.endswith('.obj'):
                    # Normalize object name by removing '_rm' suffix
                    normalized_name = obj_file.name.replace('_rm.obj', '.obj')
                    objects_by_category[category_name].add(normalized_name)
    
    return objects_by_category

def compare_folders():
    """
    Compare objects between Normalised-objects and ShapeDatabase_INFOMR-master folders.
    """
    base_path = Path("c:/Users/rob_d/Desktop/GIthub/Multimedia_Retrieval")
    
    print("Comparing objects between Normalised-objects and ShapeDatabase_INFOMR-master...\n")
    
    # Get objects from both folders
    normalised_objects = get_objects_from_folder(base_path, "Normalised-objects")
    database_objects = get_objects_from_folder(base_path, "ShapeDatabase_INFOMR-master/Original Database")
    
    # Get all categories from both folders
    all_categories = set(normalised_objects.keys()) | set(database_objects.keys())
    
    # Statistics
    total_normalised = 0
    total_database = 0
    total_missing_in_normalised = 0
    total_missing_in_database = 0
    categories_only_in_normalised = []
    categories_only_in_database = []
    
    print("=" * 80)
    print("DETAILED COMPARISON BY CATEGORY")
    print("=" * 80)
    
    for category in sorted(all_categories):
        norm_objs = normalised_objects.get(category, set())
        db_objs = database_objects.get(category, set())
        
        total_normalised += len(norm_objs)
        total_database += len(db_objs)
        
        print(f"\nCategory: {category}")
        print(f"  Normalised-objects: {len(norm_objs)} files")
        print(f"  ShapeDatabase: {len(db_objs)} files")
        
        # Check if category exists in both folders
        if category not in normalised_objects:
            categories_only_in_database.append(category)
            print(f"  ⚠️  Category '{category}' ONLY exists in ShapeDatabase")
            continue
        elif category not in database_objects:
            categories_only_in_normalised.append(category)
            print(f"  ⚠️  Category '{category}' ONLY exists in Normalised-objects")
            continue
        
        # Find missing objects
        missing_in_normalised = db_objs - norm_objs
        missing_in_database = norm_objs - db_objs
        
        total_missing_in_normalised += len(missing_in_normalised)
        total_missing_in_database += len(missing_in_database)
        
        if missing_in_normalised:
            print(f"  📥 Missing in Normalised-objects ({len(missing_in_normalised)}):")
            for obj in sorted(missing_in_normalised):
                print(f"    - {obj}")
        
        if missing_in_database:
            print(f"  📤 Missing in ShapeDatabase ({len(missing_in_database)}):")
            for obj in sorted(missing_in_database):
                print(f"    - {obj}")
        
        if not missing_in_normalised and not missing_in_database:
            print(f"  ✅ All objects match!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total objects in Normalised-objects: {total_normalised}")
    print(f"Total objects in ShapeDatabase: {total_database}")
    print(f"Difference: {abs(total_normalised - total_database)} objects")
    print()
    print(f"Objects missing in Normalised-objects: {total_missing_in_normalised}")
    print(f"Objects missing in ShapeDatabase: {total_missing_in_database}")
    
    if categories_only_in_normalised:
        print(f"\nCategories only in Normalised-objects ({len(categories_only_in_normalised)}):")
        for cat in categories_only_in_normalised:
            print(f"  - {cat}")
    
    if categories_only_in_database:
        print(f"\nCategories only in ShapeDatabase ({len(categories_only_in_database)}):")
        for cat in categories_only_in_database:
            print(f"  - {cat}")

def quick_summary():
    """
    Show a quick summary without detailed object lists.
    """
    base_path = Path("c:/Users/rob_d/Desktop/GIthub/Multimedia_Retrieval")
    
    print("Quick summary of object counts by category...\n")
    
    # Get objects from both folders
    normalised_objects = get_objects_from_folder(base_path, "Normalised-objects")
    database_objects = get_objects_from_folder(base_path, "ShapeDatabase_INFOMR-master/Original Database")
    
    # Get all categories from both folders
    all_categories = set(normalised_objects.keys()) | set(database_objects.keys())
    
    print(f"{'Category':<25} {'Normalised':<12} {'Database':<12} {'Difference':<12}")
    print("-" * 65)
    
    total_normalised = 0
    total_database = 0
    
    for category in sorted(all_categories):
        norm_count = len(normalised_objects.get(category, set()))
        db_count = len(database_objects.get(category, set()))
        difference = norm_count - db_count
        
        total_normalised += norm_count
        total_database += db_count
        
        status = ""
        if norm_count == 0:
            status = "(Only in DB)"
        elif db_count == 0:
            status = "(Only in Norm)"
        elif difference != 0:
            status = f"({difference:+d})"
        
        print(f"{category:<25} {norm_count:<12} {db_count:<12} {status:<12}")
    
    print("-" * 65)
    print(f"{'TOTAL':<25} {total_normalised:<12} {total_database:<12} {total_normalised - total_database:+d}")

if __name__ == "__main__":
    print("Object Comparison Script")
    print("=======================")
    print("1. Quick summary (counts only)")
    print("2. Detailed comparison (with missing object lists)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_summary()
    elif choice == "2":
        compare_folders()
    else:
        print("Invalid choice. Running quick summary...")
        quick_summary()
