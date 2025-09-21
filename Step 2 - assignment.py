import os
import pandas as pd
import matplotlib.pyplot as plt

# Analyze a single .obj file for vertices, faces, face types, and bounding box
def analyze_shape(file_path):
    vertices = []
    face_vertex_counts = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  # Vertex line
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':  # Face line
                face_vertex_counts.append(len(parts) - 1)

    # Get class from parent folder
    shape_class = os.path.basename(os.path.dirname(file_path))
    num_vertices = len(vertices)
    num_faces = len(face_vertex_counts)

    # Determine face type
    face_types = set()
    for count in face_vertex_counts:
        if count == 3:
            face_types.add("triangles")
        elif count == 4:
            face_types.add("quads")
        else:
            face_types.add("other")
    face_type_str = " and ".join(sorted(face_types)) if face_types else "unknown"

    # Calculate bounding box
    if vertices:
        xs, ys, zs = zip(*vertices)
        bounding_box = {
            "min_x": min(xs), "max_x": max(xs),
            "min_y": min(ys), "max_y": max(ys),
            "min_z": min(zs), "max_z": max(zs)
        }
    else:
        bounding_box = {k: None for k in ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]}

    return {
        "filename": os.path.basename(file_path),
        "class": shape_class,
        "vertices": num_vertices,
        "faces": num_faces,
        "face_type": face_type_str,
        "bounding_box": bounding_box
    }

# Find all .obj files in a folder
def find_obj_files(folder_path):
    obj_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    return sorted(obj_files)

# Main analysis and plotting
def main():
    folder_name = 'ShapeDatabase_INFOMR-master'
    if not os.path.isdir(folder_name):
        print(f"Folder '{folder_name}' not found.")
        return

    obj_files = find_obj_files(folder_name)
    if not obj_files:
        print("No .obj files found.")
        return

    # Analyze all files and create csv
    results = [analyze_shape(fp) for fp in obj_files]
    df = pd.DataFrame(results)
    df.to_csv('shape_database_analysis.csv', index=False)
    print("Analysis saved to 'shape_database_analysis.csv'.")

    # Print averages
    print(f"Average vertices: {df['vertices'].mean():.2f}")
    print(f"Average faces: {df['faces'].mean():.2f}")

    # Plot histograms
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    df['vertices'].hist(bins=20, color='skyblue')
    plt.title('Vertex Count Distribution')
    plt.subplot(1, 2, 2)
    df['faces'].hist(bins=20, color='lightcoral')
    plt.title('Face Count Distribution')
    plt.tight_layout()
    plt.savefig('vertex_face_histograms.png')
    print("Histograms saved.")

    # Plot class bar chart
    plt.figure(figsize=(10, 6))
    df['class'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Shapes per Class')
    plt.tight_layout()
    plt.savefig('shape_class_bar_chart.png')
    print("Bar chart saved.")

if __name__ == "__main__":
    main()
