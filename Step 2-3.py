import os
import shutil
import trimesh

# ----- Instellingen -----
input_folder = "ShapeDatabase_INFOMR-master/ShapeDatabase_INFOMR-master"  # originele map
output_folder = "copy5000/copy5000"  # nieuwe map voor kopieën
target_vertices = 5000

# Loop door alle bestanden en submappen
for root, dirs, files in os.walk(input_folder):
    # Maak dezelfde submapstructuur in de output map
    relative_path = os.path.relpath(root, input_folder)
    output_path = os.path.join(output_folder, relative_path)
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        if file.lower().endswith(".obj"):
            input_file = os.path.join(root, file)
            output_file_original = os.path.join(output_path, file)

            # --- Kopieer origineel ---
            shutil.copy2(input_file, output_file_original)

            # --- Pas mesh aan en sla op als _copy.obj ---
            mesh = trimesh.load(input_file)
            current_vertices = len(mesh.vertices)

            # Subdivide als te weinig vertices
            if current_vertices < target_vertices:
                while len(mesh.vertices) < target_vertices/2:
                    mesh = mesh.subdivide()

            # Simplify als te veel vertices
            elif current_vertices > target_vertices:
                factor = 1 - target_vertices / current_vertices
                mesh = mesh.simplify_quadric_decimation(factor)

            # Output bestand met _copy
            name, ext = os.path.splitext(file)
            output_file_copy = os.path.join(output_path, f"{name}_copy{ext}")
            mesh.export(output_file_copy)

            print(f"Original: {output_file_original} | Copy: {output_file_copy} ({len(mesh.vertices)} vertices)")

print("Klaar! Alle bestanden gekopieerd en aangepaste kopieën aangemaakt.")
