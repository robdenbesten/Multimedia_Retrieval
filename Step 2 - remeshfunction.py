import os
import shutil
import pymeshlab as ml

# ----- Instellingen -----
input_folder = "ShapeDatabase_INFOMR-master/Original Database/PlantIndoors"  # originele map
output_folder = "copy5000/copy5000/plant"  # nieuwe map voor kopieën

TARGET_VERTICES = 5000
Error = False

# ----- Functie -----
def remeshObject(input_file, output_file_copy):
    global success
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file)

    # Clean mesh to ensure manifoldness
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    # ms.apply_filter("meshing_close_holes", maxholesize=10)  # Optional: try to close small holes

    def decreaseVertices():
        try:
            print(f"Too many vertices, current number: {ms.current_mesh().vertex_number()} (f {ms.current_mesh().face_number()})")
            estimated_amount_faces = int(ms.current_mesh().face_number() * (TARGET_VERTICES / ms.current_mesh().vertex_number()))
            print(estimated_amount_faces)
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse",
                targetfacenum=estimated_amount_faces,
                qualitythr=0.5,
                preservenormal=True,
                preserveboundary=True,
                preservetopology=True,
                optimalplacement=True,
                autoclean=True
            )
            return True  # success
        except ml.PyMeshLabException as e:
            print(f"Error with Quadratic Edge Collapse Decimation: vertices {ms.current_mesh().vertex_number()} ({e})")
            return False  # failure

    def increaseVertices():
        try:
            print(f"Too little vertices, current number: {ms.current_mesh().vertex_number()} (f {ms.current_mesh().face_number()})")
            ms.apply_filter(
                "meshing_surface_subdivision_midpoint",
                iterations=1
            )
            return True  # success
        except ml.PyMeshLabException as e:
            print(f"Error with Surface Subdivision Midpoint: vertices {ms.current_mesh().vertex_number()} ({e})")
            return False  # failure

    # Remeshing logic
    if ms.current_mesh().vertex_number() < TARGET_VERTICES:
        counter = 0
        while (ms.current_mesh().vertex_number() < TARGET_VERTICES) & (counter < 10):
            counter += 1
            success = increaseVertices()
            if not success:
                print("Subdivision failed, skipping further attempts.")
                break
        if success:
            while (ms.current_mesh().vertex_number() > TARGET_VERTICES) & (counter < 10):
                counter += 1
                success = decreaseVertices()
                if not success:
                    print("Simplifying failed, skipping further attempts.")
                    break
    else:
        counter = 0
        while (ms.current_mesh().vertex_number() > TARGET_VERTICES) & (counter < 10):
            counter += 1
            success = decreaseVertices()
            if not success:
                print("Simplifying failed, skipping further attempts.")
                break

    # Save the result
    try:
        ms.save_current_mesh(output_file_copy)
        print(f"{output_file_copy}: final vertices: {ms.current_mesh().vertex_number()}")
        print(" ")
    except ml.PyMeshLabException as e:
        print(f"Could not save mesh: {e}")
        print(" ")


# remeshObject("D00921.obj", "test.obj")


# ----- Loop door alle bestanden -----
for root, dirs, files in os.walk(input_folder):
    # Maak dezelfde submapstructuur in de output map
    relative_path = os.path.relpath(root, input_folder)
    output_path = os.path.join(output_folder, relative_path)
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        if file.lower().endswith(".obj"):
            input_file = os.path.join(root, file)

            # --- Kopieer origineel ---
            # output_file_original = os.path.join(output_path, file)
            # shutil.copy2(input_file, output_file_original)

            # --- Pas mesh aan en sla op als _copy.obj ---
            name, ext = os.path.splitext(file)
            output_file_copy = os.path.join(output_path, f"{name}_copy{ext}")

            # Wrap the remeshObject call in try-except to skip errors
            try:
                remeshObject(input_file, output_file_copy)
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
                print(" ")


print("Klaar! Alle bestanden gekopieerd en aangepaste kopieën aangemaakt.")