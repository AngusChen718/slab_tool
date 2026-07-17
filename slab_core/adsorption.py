"""Adsorbate orientation and placement on validated slab candidates."""

import numpy as np
from ase.constraints import FixAtoms

from .analysis import normal_cell_height, surface_normal
from .models import BuildCandidate


def _rotation_between(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, first)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(first, trial)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _reference_axis(atoms, anchor_index):
    if len(atoms) <= 1:
        return None
    anchor = np.asarray(atoms.positions[anchor_index], dtype=float)
    others = np.delete(np.asarray(atoms.positions, dtype=float), anchor_index, axis=0)
    vector = np.mean(others, axis=0) - anchor
    if np.linalg.norm(vector) <= 1e-10:
        # Symmetric molecules can have their centroid on the anchor.  Use the
        # vector to the farthest non-anchor atom as a deterministic axis.
        vectors = others - anchor
        vector = vectors[int(np.argmax(np.linalg.norm(vectors, axis=1)))]
    if np.linalg.norm(vector) <= 1e-10:
        raise ValueError("無法定義吸附物的參考方向")
    return vector / np.linalg.norm(vector)


def _desired_axis(slab, tilt_deg, azimuth_deg):
    normal = surface_normal(slab)
    # ``np.asarray(CellRow)`` can be a writable view into the ASE cell.  Copy
    # before normalization so orientation math can never alter the slab cell.
    e1 = np.array(slab.cell[0], dtype=float, copy=True)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    tilt = np.deg2rad(float(tilt_deg))
    azimuth = np.deg2rad(float(azimuth_deg))
    lateral = np.cos(azimuth) * e1 + np.sin(azimuth) * e2
    value = np.cos(tilt) * normal + np.sin(tilt) * lateral
    return value / np.linalg.norm(value)


def _internal_distances(atoms):
    result = {}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            result[f"{i}-{j}"] = float(np.linalg.norm(
                np.asarray(atoms.positions[j]) - np.asarray(atoms.positions[i])
            ))
    return result


def place_adsorbate(
    slab_candidate,
    adsorbate_spec,
    site,
    height_A=2.0,
    tilt_deg=0.0,
    azimuth_deg=0.0,
    vacuum_total_A=15.0,
):
    if float(height_A) <= 0:
        raise ValueError("吸附高度必須大於 0 Å")
    if not 0 <= int(adsorbate_spec.anchor_index) < len(adsorbate_spec.atoms):
        raise ValueError("錨定原子索引超出範圍")

    slab = slab_candidate.atoms.copy()
    adsorbate = adsorbate_spec.atoms.copy()
    anchor_index = int(adsorbate_spec.anchor_index)
    reference_distances = _internal_distances(adsorbate)
    anchor = np.asarray(adsorbate.positions[anchor_index], dtype=float)
    adsorbate.translate(-anchor)

    reference_axis = _reference_axis(adsorbate, anchor_index)
    if reference_axis is not None:
        rotation = _rotation_between(
            reference_axis,
            _desired_axis(slab, tilt_deg, azimuth_deg),
        )
        adsorbate.positions[:] = np.asarray(adsorbate.positions) @ rotation.T

    normal = surface_normal(slab)
    target = np.asarray(site.cartesian, dtype=float) + float(height_A) * normal
    adsorbate.translate(target)

    # Translate the complete molecule by lattice vectors according to the
    # anchor position.  Do not wrap individual atoms, which could split a
    # molecule across opposite sides of the displayed cell.
    anchor_cartesian = np.asarray(adsorbate.positions[anchor_index])
    anchor_scaled = np.linalg.solve(np.asarray(slab.cell).T, anchor_cartesian)
    lateral_shift = np.zeros(3)
    for axis in (0, 1):
        integer = np.floor(anchor_scaled[axis] + 1e-12)
        lateral_shift -= integer * np.asarray(slab.cell[axis])
    adsorbate.translate(lateral_shift)

    slab_count = len(slab)
    combined = slab.copy()
    combined.extend(adsorbate)
    combined.set_pbc((True, True, True))
    # Preserve the requested periodic vacuum after the adsorbate increases the
    # occupied normal span.  This translates the whole system together and
    # does not change adsorption height or any internal distance.
    combined.center(vacuum=float(vacuum_total_A) / 2.0, axis=2)

    fixed_indices = list(slab_candidate.metadata.get("fixed_indices", []))
    combined.set_constraint()
    if fixed_indices:
        combined.set_constraint(FixAtoms(indices=fixed_indices))
    adsorbate_indices = list(range(slab_count, len(combined)))

    candidate_id = (
        f"ads_{adsorbate_spec.name}_{site.site_id}_"
        f"h{float(height_A):.2f}_t{float(tilt_deg):.1f}_a{float(azimuth_deg):.1f}"
    ).replace(".", "p")
    metadata = {
        **slab_candidate.metadata,
        "route": "adsorption",
        "base_candidate_id": slab_candidate.candidate_id,
        "adsorbate_name": adsorbate_spec.name,
        "adsorbate_formula": adsorbate.get_chemical_formula(),
        "adsorbate_indices": adsorbate_indices,
        "slab_indices": list(range(slab_count)),
        "anchor_local_index": anchor_index,
        "anchor_index": slab_count + anchor_index,
        "site": site.to_dict(),
        "requested_height_A": float(height_A),
        "tilt_deg": float(tilt_deg),
        "azimuth_deg": float(azimuth_deg),
        "requested_vacuum_A": float(vacuum_total_A),
        "reference_internal_distances_A": reference_distances,
        "fixed_indices": fixed_indices,
        "cell_normal_height_A": normal_cell_height(combined),
        "one_sided_adsorption": True,
    }
    return BuildCandidate(
        atoms=combined,
        candidate_id=candidate_id,
        builder="slab_core.adsorption.place_adsorbate",
        metadata=metadata,
    )
