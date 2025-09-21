import os
import shutil
import pymeshlab as ml

# ----- Instellingen -----
input_folder = "ShapeDatabase_INFOMR-master/ShapeDatabase_INFOMR-master"  # originele map
output_folder = "copy5001/copy5001"  # nieuwe map voor kopieën

TARGET_VERTS = 5000
TOLERANCE = 100
MAX_ITERS = 12

# ----- Functie -----
def remeshObject(input_file, output_file_copy):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file)

    def verts_faces():
        mesh = ms.current_mesh()
        return mesh.vertex_number(), mesh.face_number()

    for it in range(1, MAX_ITERS + 1):
        cur_v, cur_f = verts_faces()

        # Stop als binnen tolerantie
        if abs(cur_v - TARGET_VERTS) <= TOLERANCE:
            break

        # Te weinig vertices -> refine
        if cur_v < TARGET_VERTS:
            try:
                print(f"Te weinig vertices: {cur_v}")
                ms.apply_filter("meshing_surface_subdivision_loop", iterations=1)
            except ml.PyMeshLabException as e:
                print(f"Iter {it}: Kan subdivide niet toepassen ({e}), overslaan")
                break  # optioneel: stop subdivide als het niet kan
        # Te veel vertices -> decimeer
        else:
            est_target_faces = max(4, int(cur_f * (TARGET_VERTS / cur_v))) if cur_v > 0 else TARGET_VERTS
            try:
                print(f"Te veel vertices: {cur_v}")
                ms.apply_filter(
                    "meshing_decimation_quadric_edge_collapse",
                    targetfacenum=est_target_faces,
                    preservenormal=True,
                    preserveboundary=True,
                    optimalplacement=True
                )
            except ml.PyMeshLabException as e:
                print(f"Iter {it}: Kan decimeer niet toepassen ({e}), overslaan")
                break  # stop decimeer als het niet kan

    # Sla het resultaat op
    try:
        ms.save_current_mesh(output_file_copy)
        final_v, _ = verts_faces()
        print(f"{output_file_copy}: final vertices: {final_v}")
    except ml.PyMeshLabException as e:
        print(f"Kan mesh niet opslaan: {e}")

# remeshObject("m1345.obj", "test.obj")

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
            output_file_original = os.path.join(output_path, file)
            shutil.copy2(input_file, output_file_original)

            # --- Pas mesh aan en sla op als _copy.obj ---
            name, ext = os.path.splitext(file)
            output_file_copy = os.path.join(output_path, f"{name}_copy{ext}")
            remeshObject(input_file, output_file_copy)

print("Klaar! Alle bestanden gekopieerd en aangepaste kopieën aangemaakt.")
