"""Physics-oriented validation for slab plus adsorbate candidates."""

import numpy as np
from ase.data import covalent_radii

from .analysis import find_atomic_planes, normal_cell_height, surface_normal
from .models import ValidationReport


def _cross_distances(atoms, slab_indices, adsorbate_indices):
    records = []
    for slab_index in slab_indices:
        for adsorbate_index in adsorbate_indices:
            distance = float(atoms.get_distance(
                slab_index, adsorbate_index, mic=True
            ))
            radii_sum = float(
                covalent_radii[atoms.numbers[slab_index]]
                + covalent_radii[atoms.numbers[adsorbate_index]]
            )
            records.append((slab_index, adsorbate_index, distance, radii_sum))
    return records


def _periodic_adsorbate_image_distance(atoms, adsorbate_indices):
    a = np.asarray(atoms.cell[0], dtype=float)
    b = np.asarray(atoms.cell[1], dtype=float)
    positions = np.asarray(atoms.positions)[adsorbate_indices]
    minimum = float("inf")
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            if sx == 0 and sy == 0:
                continue
            shift = sx * a + sy * b
            for first in positions:
                for second in positions:
                    minimum = min(minimum, float(np.linalg.norm(second + shift - first)))
    return minimum


def _internal_distances(atoms, indices):
    result = {}
    for local_i, atom_i in enumerate(indices):
        for local_j in range(local_i + 1, len(indices)):
            atom_j = indices[local_j]
            result[f"{local_i}-{local_j}"] = float(np.linalg.norm(
                np.asarray(atoms.positions[atom_j]) - np.asarray(atoms.positions[atom_i])
            ))
    return result


def validate_adsorption_candidate(
    candidate,
    minimum_vacuum_A=10.0,
    warning_image_distance_A=4.0,
):
    atoms = candidate.atoms
    metadata = candidate.metadata
    slab_indices = [int(v) for v in metadata.get("slab_indices", [])]
    adsorbate_indices = [int(v) for v in metadata.get("adsorbate_indices", [])]
    errors = []
    warnings = []
    checks = {}

    valid_partition = (
        bool(slab_indices)
        and bool(adsorbate_indices)
        and set(slab_indices).isdisjoint(adsorbate_indices)
        and sorted(slab_indices + adsorbate_indices) == list(range(len(atoms)))
    )
    checks["valid_atom_partition"] = valid_partition
    if not valid_partition:
        return ValidationReport(
            status="failed",
            errors=["Slab／吸附物原子索引分區無效"],
            checks=checks,
        )

    cross = _cross_distances(atoms, slab_indices, adsorbate_indices)
    minimum_cross = min(item[2] for item in cross)
    severe = [
        item for item in cross
        if item[2] < max(0.65, 0.55 * item[3])
    ]
    checks["no_adsorbate_surface_overlap"] = not severe
    if severe:
        errors.append(
            f"吸附物與表面有 {len(severe)} 組嚴重重疊；"
            f"最短距離 {minimum_cross:.4f} Å"
        )

    reference = metadata.get("reference_internal_distances_A", {})
    current = _internal_distances(atoms, adsorbate_indices)
    deviations = [
        abs(float(current[key]) - float(value))
        for key, value in reference.items()
        if key in current
    ]
    maximum_deviation = max(deviations) if deviations else 0.0
    checks["adsorbate_geometry_preserved"] = maximum_deviation <= 1e-6
    if maximum_deviation > 1e-6:
        errors.append(
            f"放置過程改變吸附物內部距離，最大偏差 {maximum_deviation:.3e} Å"
        )

    normal = surface_normal(atoms)
    slab_heights = np.asarray(atoms.positions)[slab_indices] @ normal
    adsorbate_heights = np.asarray(atoms.positions)[adsorbate_indices] @ normal
    top_height = float(np.max(slab_heights))
    anchor_index = int(metadata["anchor_index"])
    measured_height = float(np.dot(atoms.positions[anchor_index], normal) - top_height)
    requested_height = float(metadata.get("requested_height_A", measured_height))
    checks["anchor_height_matches"] = abs(measured_height - requested_height) <= 1e-5
    if measured_height <= 0:
        errors.append("錨定原子位於頂層表面以下")
    elif abs(measured_height - requested_height) > 1e-5:
        errors.append(
            f"實測吸附高度 {measured_height:.4f} Å 與設定 "
            f"{requested_height:.4f} Å 不一致"
        )

    occupied_span = float(np.max(np.concatenate([slab_heights, adsorbate_heights])) - np.min(slab_heights))
    remaining_vacuum = normal_cell_height(atoms) - occupied_span
    checks["sufficient_periodic_vacuum"] = remaining_vacuum >= float(minimum_vacuum_A)
    if remaining_vacuum < 5.0:
        errors.append(f"吸附後週期真空只剩 {remaining_vacuum:.2f} Å")
    elif remaining_vacuum < float(minimum_vacuum_A):
        warnings.append(
            f"吸附後週期真空為 {remaining_vacuum:.2f} Å；"
            "建議至少以 10 Å 作為收斂測試起點"
        )

    image_distance = _periodic_adsorbate_image_distance(atoms, adsorbate_indices)
    checks["no_severe_adsorbate_image_overlap"] = image_distance >= 2.0
    if image_distance < 2.0:
        errors.append(
            f"吸附物與其平面週期影像距離只有 {image_distance:.2f} Å"
        )
    elif image_distance < float(warning_image_distance_A):
        warnings.append(
            f"吸附物週期影像最近距離 {image_distance:.2f} Å；"
            "需檢查覆蓋率與側向交互作用收斂"
        )

    fixed_indices = set(int(v) for v in metadata.get("fixed_indices", []))
    checks["adsorbate_not_fixed"] = fixed_indices.isdisjoint(adsorbate_indices)
    if not checks["adsorbate_not_fixed"]:
        errors.append("吸附物原子不應被標記為固定")

    slab_only = atoms[slab_indices]
    slab_planes = find_atomic_planes(slab_only)
    top_plane_atoms = len(slab_planes[-1]) if slab_planes else 0
    coverage = 1.0 / top_plane_atoms if top_plane_atoms else None

    if metadata.get("one_sided_adsorption", True):
        warnings.append(
            "單面吸附會形成非對稱週期模型；後續 DFT 應評估 dipole correction"
        )
    slab_symbols = {atoms[index].symbol for index in slab_indices}
    magnetic_elements = sorted(slab_symbols.intersection({"Fe", "Co", "Ni"}))
    if magnetic_elements:
        warnings.append(
            "表面含可能具磁性的元素 " + ", ".join(magnetic_elements)
            + "；後續 DFT 必須明確設定自旋極化與初始磁矩"
        )
    if metadata.get("polar") is True:
        warnings.append("基底 termination 可能具有極性；吸附能比較前需檢查偶極處理")
    if metadata.get("symmetrized_by_atom_removal", False):
        warnings.append(
            "基底 termination 曾由對稱化移除表面原子；請確認化學計量與研究目的"
        )

    metrics = {
        "builder": candidate.builder,
        "candidate_id": candidate.candidate_id,
        "total_atom_count": len(atoms),
        "slab_atom_count": len(slab_indices),
        "adsorbate_atom_count": len(adsorbate_indices),
        "adsorbate_formula": metadata.get("adsorbate_formula"),
        "site_id": metadata.get("site", {}).get("site_id"),
        "site_type": metadata.get("site", {}).get("site_type"),
        "requested_height_A": requested_height,
        "measured_anchor_height_A": measured_height,
        "minimum_adsorbate_surface_distance_A": minimum_cross,
        "remaining_vacuum_A": remaining_vacuum,
        "adsorbate_image_distance_A": image_distance,
        "maximum_internal_distance_change_A": maximum_deviation,
        "coverage_ML_approx": coverage,
        "tilt_deg": metadata.get("tilt_deg"),
        "azimuth_deg": metadata.get("azimuth_deg"),
    }
    return ValidationReport(
        status="passed" if not errors else "failed",
        errors=errors,
        warnings=warnings,
        metrics=metrics,
        checks=checks,
    )
