import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def analyze_shape(file_path):
    """
    Parses a single .obj shape file to analyze its properties without external libraries.

    Args:
        file_path (str): The full path to the .obj file.

    Returns:
        dict: A dictionary containing the shape's analysis results.
              Returns None if the file cannot be processed.
    """
    try:
        vertices = []
        face_vertex_counts = []

        # Open and read the file line by line
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                # Check for vertex lines (lines starting with 'v')
                if parts[0] == 'v' and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

                # Check for face lines (lines starting with 'f')
                elif parts[0] == 'f' and len(parts) >= 4:
                    # Count the number of vertices in the face
                    face_vertex_counts.append(len(parts) - 1)

        # Get the shape's class from the parent folder name
        shape_class = os.path.basename(os.path.dirname(file_path))

        # Get the number of vertices and faces
        num_vertices = len(vertices)
        num_faces = len(face_vertex_counts)

        # Analyze face types
        face_types = set()
        for count in face_vertex_counts:
            if count == 3:
                face_types.add("triangles")
            elif count == 4:
                face_types.add("quads")
            else:
                face_types.add("other")

        # Determine the type of faces as a readable string
        if len(face_types) > 1:
            face_type_str = "mix of " + " and ".join(sorted(list(face_types)))
        else:
            face_type_str = list(face_types)[0] if face_types else "unknown"

        # Calculate the axis-aligned 3D bounding box
        if not vertices:
            bounding_box = {
                "min_x": None, "max_x": None,
                "min_y": None, "max_y": None,
                "min_z": None, "max_z": None
            }
        else:
            min_x = min(v[0] for v in vertices)
            max_x = max(v[0] for v in vertices)
            min_y = min(v[1] for v in vertices)
            max_y = max(v[1] for v in vertices)
            min_z = min(v[2] for v in vertices)
            max_z = max(v[2] for v in vertices)
            bounding_box = {
                "min_x": min_x, "max_x": max_x,
                "min_y": min_y, "max_y": max_y,
                "min_z": min_z, "max_z": max_z
            }

        return {
            "filename": os.path.basename(file_path),
            "class": shape_class,
            "vertices": num_vertices,
            "faces": num_faces,
            "face_type": face_type_str,
            "bounding_box": bounding_box
        }

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
        return None


def find_obj_files(folder_path):
    """
    Recursively walks through a directory to find all .obj files.
    """
    obj_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    obj_files.sort()
    return obj_files


def main():
    """
    Main function to run the shape analysis tool.
    """
    folder_name = 'ShapeDatabase_INFOMR-master'

    if not os.path.isdir(folder_name):
        print(f"Error: The folder '{folder_name}' does not exist.")
        print("Please ensure you have the 'ShapeDatabase_INFOMR-master' folder in the same directory as this script.")
        return

    obj_files = find_obj_files(folder_name)

    if not obj_files:
        print(f"No .obj files found in '{folder_name}'.")
        return

    print(f"Found {len(obj_files)} .obj files. Starting analysis...")

    analysis_results = []
    for file_path in obj_files:
        analysis_data = analyze_shape(file_path)
        if analysis_data:
            analysis_results.append(analysis_data)

    df = pd.DataFrame(analysis_results)

    output_csv = 'shape_database_analysis.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nAnalysis results saved to '{output_csv}'.")

    avg_vertices = df['vertices'].mean()
    avg_faces = df['faces'].mean()

    print("\nStatistical Analysis:")
    print(f"Average number of vertices: {avg_vertices:.2f}")
    print(f"Average number of faces: {avg_faces:.2f}")

    if not df.empty:
        q1_v, q3_v = df['vertices'].quantile([0.25, 0.75])
        iqr_v = q3_v - q1_v
        outlier_low_v = q1_v - 1.5 * iqr_v
        outlier_high_v = q3_v + 1.5 * iqr_v

        vertex_outliers = df[(df['vertices'] < outlier_low_v) | (df['vertices'] > outlier_high_v)]
        if not vertex_outliers.empty:
            print(f"\nSignificant outliers in vertex count found:")
            print(vertex_outliers[['filename', 'vertices', 'faces']].to_string(index=False))

        q1_f, q3_f = df['faces'].quantile([0.25, 0.75])
        iqr_f = q3_f - q1_f
        outlier_low_f = q1_f - 1.5 * iqr_f
        outlier_high_f = q3_f + 1.5 * iqr_f

        face_outliers = df[(df['faces'] < outlier_low_f) | (df['faces'] > outlier_high_f)]
        if not face_outliers.empty:
            print(f"\nSignificant outliers in face count found:")
            print(face_outliers[['filename', 'vertices', 'faces']].to_string(index=False))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    df['vertices'].hist(bins=20, edgecolor='black', color='skyblue')
    plt.title('Distribution of Vertex Counts', fontsize=14)
    plt.xlabel('Number of Vertices', fontsize=12)
    plt.ylabel('Number of Shapes', fontsize=12)
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(1, 2, 2)
    df['faces'].hist(bins=20, edgecolor='black', color='lightcoral')
    plt.title('Distribution of Face Counts', fontsize=14)
    plt.xlabel('Number of Faces', fontsize=12)
    plt.ylabel('Number of Shapes', fontsize=12)
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.savefig('vertex_face_histograms.png')
    print("\nHistograms saved as 'vertex_face_histograms.png'.")

    plt.figure(figsize=(10, 6))
    class_counts = df['class'].value_counts().sort_index()
    class_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Number of Shapes per Class', fontsize=16)
    plt.xlabel('Shape Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('shape_class_bar_chart.png')
    print("Bar chart saved as 'shape_class_bar_chart.png'.")


if __name__ == "__main__":
    main()