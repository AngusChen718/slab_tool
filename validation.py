import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list

from .analysis import (
    axis_gap_analysis,
    find_atomic_planes,
    normal_cell_height,
    plane_compositions,
    surface_normal,
)
from .models import ValidationReport


def minimum_periodic_distance(atoms):
    _, _, distances = neighbor_list(
        "ijd", atoms, cutoff=natural_cutoffs(atoms, mult=2.0)
    )
    return float(np.min(distances)) if len(distances) else float("nan")


def duplicate_atom_pairs(atoms, tolerance=0.10):
    i_values, j_values, distances = neighbor_list(
        "ijd", atoms, cutoff=float(tolerance), self_interaction=False
    )
    pairs = {
        tuple(sorted((int(i), int(j))))
        for i, j, distance in zip(i_values, j_values, distances)
        if i != j and distance < tolerance
    }
    return sorted(pairs)


def severe_overlap_pairs(atoms, multiplier=0.60):
    i_values, j_values, distances = neighbor_list(
        "ijd", atoms, cutoff=natural_cutoffs(atoms, mult=multiplier)
    )
    pairs = {}
    for i, j, distance in zip(i_values, j_values, distances):
        if i == j:
            continue
        pair = tuple(sorted((int(i), int(j))))
        pairs[pair] = min(float(distance), pairs.get(pair, float("inf")))
    return pairs


def _plane_heights(atoms, planes):
    normal = surface_normal(atoms)
    heights = np.asarray(atoms.positions) @ normal
    return np.array([float(np.mean(heights[p])) for p in planes])


def validate_candidate(
    candidate,
    requested_vacuum,
    expected_atomic_layers=None,
    input_minimum_distance=None,
    strict_standard=False,
    vacuum_tolerance=0.25,
):
    atoms = candidate.atoms
    errors = []
    warnings = []
    checks = {}

    if len(atoms) == 0:
        return ValidationReport(
            status="failed", errors=["輸出結構沒有原子"]
        )

    volume = float(abs(atoms.get_volume()))
    area = float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))
    if volume <= 1e-10:
        errors.append("晶胞體積無效")
    if area <= 1e-10:
        errors.append("表面積無效")

    planes = find_atomic_planes(atoms)
    compositions = plane_compositions(atoms, planes)
    plane_counts = [len(plane) for plane in planes]
    heights = _plane_heights(atoms, planes)
    thickness = float(np.ptp(heights)) if len(heights) else 0.0
    vacuum = axis_gap_analysis(atoms, 2)["gap_A"]
    inplane_x = axis_gap_analysis(atoms, 0)
    inplane_y = axis_gap_analysis(atoms, 1)
    minimum_distance = minimum_periodic_distance(atoms)
    duplicates = duplicate_atom_pairs(atoms)
    overlaps = severe_overlap_pairs(atoms)

    checks["valid_cell"] = volume > 1e-10 and area > 1e-10
    checks["no_duplicate_atoms"] = not duplicates
    checks["no_severe_overlaps"] = not overlaps
    checks["vacuum_matches_request"] = (
        abs(vacuum - float(requested_vacuum)) <= vacuum_tolerance
    )

    if duplicates:
        errors.append(f"發現 {len(duplicates)} 組週期重複原子")
    if overlaps:
        shortest_overlap = min(overlaps.values())
        errors.append(
            f"發現 {len(overlaps)} 組疑似嚴重重疊；最短 {shortest_overlap:.4f} Å"
        )
    if abs(vacuum - float(requested_vacuum)) > vacuum_tolerance:
        errors.append(
            f"實測真空 {vacuum:.4f} Å 與設定 {requested_vacuum:.4f} Å 不一致"
        )

    for label, gap in (("x", inplane_x), ("y", inplane_y)):
        if gap["gap_A"] >= 5.0 and gap["ratio"] >= 0.25:
            message = (
                f"{label} 方向出現 {gap['gap_A']:.2f} Å 大空白，"
                "可能把舊真空旋轉到表面內方向"
            )
            if strict_standard:
                errors.append(message)
            else:
                warnings.append(message)

    if len(heights) >= 3:
        spacings = np.diff(heights)
        positive = spacings[spacings > 1e-6]
        if len(positive):
            median_spacing = float(np.median(positive))
            largest_spacing = float(np.max(positive))
            if largest_spacing >= 3.5 and largest_spacing > 2.5 * median_spacing:
                errors.append(
                    f"Slab 內部出現 {largest_spacing:.2f} Å 異常大層間距，"
                    "可能包含多個分離區塊"
                )

    if expected_atomic_layers is not None:
        if len(planes) != int(expected_atomic_layers):
            errors.append(
                f"實際原子平面 {len(planes)} 與要求 {expected_atomic_layers} 不一致"
            )
        checks["atomic_layer_count_matches"] = (
            len(planes) == int(expected_atomic_layers)
        )

    if strict_standard:
        equal_counts = len(set(plane_counts)) <= 1
        equal_compositions = all(comp == compositions[0] for comp in compositions[1:])
        checks["complete_standard_planes"] = equal_counts and equal_compositions
        if not equal_counts:
            errors.append(
                "標準單元素表面的各層原子數不一致："
                + ", ".join(str(v) for v in plane_counts)
            )
        if not equal_compositions:
            errors.append("標準單元素表面的各層元素組成不一致")

    if input_minimum_distance is not None and np.isfinite(minimum_distance):
        ratio = minimum_distance / float(input_minimum_distance)
        checks["nearest_neighbor_preserved"] = 0.80 <= ratio <= 1.20
        if ratio < 0.80:
            errors.append(
                f"輸出最短距離只有 bulk 的 {ratio:.1%}，可能發生非物理壓縮"
            )
        elif ratio > 1.20:
            warnings.append(
                f"輸出最短距離為 bulk 的 {ratio:.1%}，請確認表面晶胞"
            )

    fixed_indices = set(candidate.metadata.get("fixed_indices", []))
    fixed_plane_count = int(candidate.metadata.get("fixed_plane_count", 0))
    expected_fixed = {
        index
        for plane in planes[:fixed_plane_count]
        for index in plane
    }
    checks["fixed_complete_planes"] = fixed_indices == expected_fixed
    if fixed_indices != expected_fixed:
        errors.append("Selective Dynamics 沒有固定完整的底部原子平面")

    if candidate.metadata.get("route") == "general":
        if not candidate.metadata.get("symmetric", False):
            warnings.append("上下表面不對稱；後續 DFT 可能需要 dipole correction")
        if candidate.metadata.get("symmetrized_by_atom_removal", False):
            warnings.append(
                "此對稱候選由 pymatgen 移除部分表面原子產生；"
                "可能改變化學計量，必須人工確認 termination"
            )
        if (
            candidate.metadata.get("top_composition")
            != candidate.metadata.get("bottom_composition")
        ):
            warnings.append(
                "上下 termination 組成不同；若材料具有形式電荷，"
                "需額外評估極性與表面偶極"
            )
        if candidate.metadata.get("polar") is True:
            warnings.append("pymatgen 判定此 termination 可能具有表面偶極／極性")

    metrics = {
        "builder": candidate.builder,
        "candidate_id": candidate.candidate_id,
        "atom_count": len(atoms),
        "atomic_plane_count": len(planes),
        "atoms_per_plane": plane_counts,
        "plane_compositions": compositions,
        "surface_area_A2": area,
        "cell_normal_height_A": normal_cell_height(atoms),
        "slab_thickness_A": thickness,
        "vacuum_A": vacuum,
        "minimum_distance_A": minimum_distance,
        "duplicate_pair_count": len(duplicates),
        "severe_overlap_pair_count": len(overlaps),
        "top_composition": compositions[-1] if compositions else {},
        "bottom_composition": compositions[0] if compositions else {},
        "fixed_plane_count": fixed_plane_count,
        "fixed_atom_count": len(fixed_indices),
        **candidate.metadata,
    }
    status = "passed" if not errors else "failed"
    return ValidationReport(
        status=status,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
        checks=checks,
    )
