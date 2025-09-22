
import os
import math
import trimesh

def normalising_mesh(mesh_path, out_path=None):
    try:
        mesh = trimesh.load_mesh(mesh_path)
    except Exception as e:
        print(f"FAILED to load: {mesh_path} -> {e}")
        return False

    if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
        print(f"SKIP empty/unreadable mesh: {mesh_path}")
        return False

    # original prints and transforms kept
    print(math.dist([0,0,0], mesh.centroid))
    print(mesh.bounds[1] - mesh.bounds[0])
    centralized = mesh.centroid
    mesh.apply_translation(-centralized)
    afbaken_box = mesh.bounds
    grootte = afbaken_box[1] - afbaken_box[0]
    langste = grootte.max()
    if langste == 0:
        print(f"SKIP invalid bbox: {mesh_path}")
        return False
    scale = 1.0 / langste
    mesh.apply_scale(scale)
    print(math.dist([0, 0, 0], mesh.centroid))
    print(mesh.bounds[1] - mesh.bounds[0])

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            mesh.export(out_path)
        except Exception as e:
            print(f"FAILED to export {out_path}: {e}")
            return False
        return True
    else:
        mesh.show()
        return True

def normalize_database(src_root='ShapeDatabase_INFOMR-master/Original Database',
                       out_root='ShapeDatabase_INFOMR-master/Normalised_models'):
    src_root = os.path.abspath(src_root)
    out_root = os.path.abspath(out_root)

    files = []
    for root, _, filenames in os.walk(src_root):
        # skip any files already in the output folder (defensive)
        if os.path.commonpath([os.path.abspath(root), out_root]) == out_root:
            continue
        for fn in filenames:
            if fn.lower().endswith('.obj'):
                in_path = os.path.join(root, fn)
                rel_dir = os.path.relpath(root, src_root)
                out_path = os.path.join(out_root, rel_dir, fn)
                files.append((in_path, out_path))

    if not files:
        print("No .obj files found to process.")
        return

    print(f"Found {len(files)} .obj files. Output: {out_root}")

    processed = 0
    skipped = 0
    for i, (in_path, out_path) in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {in_path}")
        ok = normalising_mesh(in_path, out_path)
        if ok:
            processed += 1
        else:
            skipped += 1

    print(f"Done. Processed: {processed}, Skipped/Failed: {skipped}")

if __name__ == '__main__':
    normalize_database()
