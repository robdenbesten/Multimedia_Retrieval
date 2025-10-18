import os
import shutil
import pymeshlab as ml

# ----- Settings -----
INPUT_FOLDER = 'ShapeDatabase_INFOMR-master/Original Database'
OUTPUT_FOLDER = 'ShapeDatabase_INFOMR-master/copy5000'

TARGET_VERTICES = 5000
TOLERANCE = 50
MAX_ITERATIONS = 40
MIN_PROGRESS = 10
FINAL_CORRECTION_MAX = 12  # focused final attempts


def _safe_apply_filter(ms, filter_name: str, **params) -> bool:
    try:
        if ms.current_mesh().face_number() > 0:
            ms.apply_filter(filter_name, **params)
            return ms.current_mesh().face_number() > 0
        return False
    except Exception:
        return False


def _initial_aggressive_clean(ms) -> None:
    _safe_apply_filter(ms, 'meshing_remove_duplicate_vertices')
    _safe_apply_filter(ms, 'meshing_remove_duplicate_faces')
    _safe_apply_filter(ms, 'meshing_remove_null_faces')
    _safe_apply_filter(ms, 'meshing_remove_unreferenced_vertices')
    _safe_apply_filter(ms, 'meshing_repair_non_manifold_edges', method='Remove Faces')
    _safe_apply_filter(ms, 'meshing_repair_non_manifold_vertices', vertdispratio=0.0)
    _safe_apply_filter(ms, 'meshing_close_holes', maxholesize=100, newfaceselected=False)
    _safe_apply_filter(ms, 'meshing_repair_self_intersections')
    _safe_apply_filter(ms, 'meshing_triangulation')


def _maintenance_clean(ms) -> None:
    _safe_apply_filter(ms, 'meshing_remove_unreferenced_vertices')
    _safe_apply_filter(ms, 'meshing_repair_non_manifold_edges', method='Remove Faces')


def _adaptive_subdivide(ms, iters: int = 1) -> bool:
    # Try midpoint first; fallback to loop
    for _ in range(iters):
        if _safe_apply_filter(ms, 'meshing_surface_subdivision_midpoint', iterations=1):
            continue
        if not _safe_apply_filter(ms, 'meshing_surface_subdivision_loop', iterations=1):
            return False
    return True


def _smart_decimate(ms, target_v: int, aggressive: bool = False) -> bool:
    current_v = ms.current_mesh().vertex_number()
    if current_v <= target_v:
        return True

    current_f = ms.current_mesh().face_number()
    # Aim faces proportionally to vertex ratio
    target_f = max(10, int(current_f * (target_v / max(1, current_v))))

    qualitythr = 0.0 if aggressive else 0.3
    preservetopology = False if aggressive else True

    return _safe_apply_filter(
        ms,
        'meshing_decimation_quadric_edge_collapse',
        targetfacenum=target_f,
        targetperc=0.0,
        qualitythr=qualitythr,
        preserveboundary=True,
        boundaryweight=1.0,
        preservenormal=True,
        preservetopology=preservetopology,
        optimalplacement=True,
        planarquadric=True,
        qualityweight=False,
        autoclean=True
    )


def _edge_length_estimate(ms, target_v: int, scale: float = 1.5) -> float:
    bbox = ms.current_mesh().bounding_box()
    diag = bbox.diagonal()
    if diag == 0:
        return 0.0
    return (diag / (max(1, target_v) ** 0.5)) * scale


def _remesh_isotropic(ms, target_v: int, iterations: int = 15, scale: float = 1.5) -> bool:
    target_edge = _edge_length_estimate(ms, target_v, scale=scale)
    if target_edge <= 0.0:
        return False
    return _safe_apply_filter(
        ms,
        'meshing_isotropic_explicit_remeshing',
        iterations=iterations,
        adaptive=True,
        targetlen=target_edge
    )


def _tune_isotropic_to_target(ms, target_v: int, max_trials: int = 6) -> bool:
    # Binary search over targetlen to drive vertex count into tolerance
    base = _edge_length_estimate(ms, target_v, scale=1.5)
    if base <= 0.0:
        return False
    low = base / 3.0
    high = base * 3.0

    ok_any = False
    for _ in range(max_trials):
        tl = 0.5 * (low + high)
        if not _safe_apply_filter(
            ms,
            'meshing_isotropic_explicit_remeshing',
            iterations=8,
            adaptive=True,
            targetlen=tl
        ):
            break
        ok_any = True
        cv = ms.current_mesh().vertex_number()
        if abs(cv - target_v) <= TOLERANCE:
            return True
        # Too many vertices -> increase edge length
        if cv > target_v:
            low = tl
        else:
            high = tl
    return ok_any


def _precision_correction(ms, target_v: int) -> None:
    # Deterministic push into tolerance with escalation
    prev_v = -1
    stagnation = 0
    for i in range(FINAL_CORRECTION_MAX):
        cv = ms.current_mesh().vertex_number()
        if abs(cv - target_v) <= TOLERANCE:
            break

        if cv > target_v + TOLERANCE:
            aggressive = (i >= FINAL_CORRECTION_MAX // 2)
            if not _smart_decimate(ms, target_v, aggressive=aggressive):
                break
        else:
            # Decide subdivision depth based on how far we are
            factor = target_v / max(1, cv)
            iters = 2 if factor > 1.8 else 1
            if not _adaptive_subdivide(ms, iters=iters):
                break

        # Stagnation guard
        new_v = ms.current_mesh().vertex_number()
        if abs(new_v - prev_v) < max(5, MIN_PROGRESS // 2):
            stagnation += 1
        else:
            stagnation = 0
        prev_v = new_v

        if i % 3 == 2:
            _maintenance_clean(ms)

        if stagnation >= 2:
            # Try a short isotropic nudge
            _remesh_isotropic(ms, target_v, iterations=6, scale=1.3)
            stagnation = 0


def remesh_to_target_vertices(input_file: str, output_file: str, target_v: int = TARGET_VERTICES) -> None:
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file)

    mesh = ms.current_mesh()
    if mesh.vertex_number() < 10 or mesh.face_number() < 5:
        print(f'⊘ Skipping {os.path.basename(input_file)}: mesh is too small or degenerate.')
        return

    initial_v = mesh.vertex_number()
    print(f'→ Processing {os.path.basename(input_file)}: {initial_v} → {target_v} vertices')

    _initial_aggressive_clean(ms)

    iteration = 0
    last_v = -1
    stagnation_remesh_done = False

    while iteration < MAX_ITERATIONS:
        current_v = ms.current_mesh().vertex_number()
        if ms.current_mesh().face_number() < 3:
            print(f'✗ Aborting {os.path.basename(input_file)}: mesh degenerated.')
            return

        if abs(current_v - target_v) <= TOLERANCE:
            break

        if abs(current_v - last_v) < MIN_PROGRESS:
            if not stagnation_remesh_done:
                _remesh_isotropic(ms, target_v, iterations=10, scale=1.5)
                stagnation_remesh_done = True
                continue
            else:
                break

        last_v = current_v

        if current_v < target_v - TOLERANCE:
            if not _adaptive_subdivide(ms):
                break
        else:
            if not _smart_decimate(ms, target_v):
                break

        iteration += 1
        if iteration % 10 == 0:
            _maintenance_clean(ms)

    # Final precision correction and isotropic tuning if still off
    _precision_correction(ms, target_v)
    if abs(ms.current_mesh().vertex_number() - target_v) > TOLERANCE:
        _tune_isotropic_to_target(ms, target_v)

    _maintenance_clean(ms)
    _safe_apply_filter(ms, 'meshing_triangulation')

    final_v = ms.current_mesh().vertex_number()
    deviation = abs(final_v - target_v)
    status = '✓' if deviation <= TOLERANCE else '~'

    ms.save_current_mesh(output_file)
    print(f'{status} {os.path.basename(output_file)}: {final_v} vertices (Δ{deviation})')


def main():
    processed = 0
    failed = 0

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for root, _, files in os.walk(INPUT_FOLDER):
        rel_path = os.path.relpath(root, INPUT_FOLDER)
        out_path = os.path.join(OUTPUT_FOLDER, rel_path)
        os.makedirs(out_path, exist_ok=True)

        for file in files:
            if not file.lower().endswith('.obj'):
                continue

            src = os.path.join(root, file)
            name, ext = os.path.splitext(file)
            dst_processed = os.path.join(out_path, f'{name}_remeshed{ext}')

            try:
                remesh_to_target_vertices(src, dst_processed, TARGET_VERTICES)
                processed += 1
            except Exception as e:
                print(f'✗ Failed {file}: {e}')
                failed += 1

    print(f'\n{"=" * 60}')
    print(f'Complete: {processed} processed, {failed} failed')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()