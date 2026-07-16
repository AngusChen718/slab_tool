"""Stage 1 physical-geometry helpers for the Slab Builder.

This module deliberately does not classify crystal structures from coordination
numbers.  It reports measurable geometry and keeps bulk cutting separate from
existing-slab processing.
"""

from math import gcd
from functools import reduce

import numpy as np
from ase.build import surface
from ase.constraints import FixAtoms
from ase.neighborlist import natural_cutoffs, neighbor_list


def reduce_miller_index(hkl):
    values = tuple(int(v) for v in hkl)
    if values == (0, 0, 0):
        raise ValueError("Miller index 不可為 (0, 0, 0)")
    divisor = reduce(gcd, (abs(v) for v in values if v != 0))
    return tuple(v // divisor for v in values)


def surface_normal(atoms):
    a, b = np.asarray(atoms.cell[0]), np.asarray(atoms.cell[1])
    normal = np.cross(a, b)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ValueError("晶胞 a、b 向量無法定義有效表面法向")
    return normal / length


def normal_cell_height(atoms):
    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    if area <= 1e-12:
        raise ValueError("晶胞表面積必須大於零")
    return abs(atoms.get_volume()) / area


def periodic_gap_analysis(atoms, gap_threshold=3.5, ratio_threshold=0.25):
    """Measure the largest cyclic empty interval along the c direction.

    A large interval is evidence that the input may already be a slab.  It is
    intentionally a warning heuristic rather than a universal crystal label.
    """
    if len(atoms) == 0:
        raise ValueError("輸入結構沒有原子")
    scaled_c = np.mod(atoms.get_scaled_positions(wrap=False)[:, 2], 1.0)
    scaled_c.sort()
    cyclic = np.diff(np.concatenate([scaled_c, [scaled_c[0] + 1.0]]))
    largest_fraction = float(np.max(cyclic))
    height = normal_cell_height(atoms)
    largest_gap = largest_fraction * height
    occupied_height = max(height - largest_gap, 0.0)
    likely_slab = (
        largest_gap >= gap_threshold
        and largest_gap / height >= ratio_threshold
    )
    return {
        "cell_height": height,
        "largest_gap": largest_gap,
        "occupied_height": occupied_height,
        "gap_ratio": largest_gap / height,
        "likely_slab": likely_slab,
    }


def find_atomic_planes(atoms, tolerance=0.20):
    """Return atom-index groups for complete planes, ordered bottom to top."""
    normal = surface_normal(atoms)
    heights = np.asarray(atoms.positions) @ normal
    order = np.argsort(heights)
    planes = []
    for atom_index in order:
        atom_index = int(atom_index)
        if not planes:
            planes.append([atom_index])
            continue
        plane_height = float(np.mean(heights[planes[-1]]))
        if abs(float(heights[atom_index]) - plane_height) <= tolerance:
            planes[-1].append(atom_index)
        else:
            planes.append([atom_index])
    return planes


def describe_input_cell(atoms):
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    return (
        f"{atoms.get_chemical_formula()}；{len(atoms)} atoms；"
        f"a/b/c = {lengths[0]:.4f}/{lengths[1]:.4f}/{lengths[2]:.4f} Å；"
        f"α/β/γ = {angles[0]:.2f}/{angles[1]:.2f}/{angles[2]:.2f}°"
    )


def build_from_bulk(bulk, hkl, layers, vacuum_total, supercell):
    reduced_hkl = reduce_miller_index(hkl)
    slab = surface(
        bulk,
        reduced_hkl,
        layers=int(layers),
        vacuum=float(vacuum_total) / 2.0,
    )
    slab *= (int(supercell[0]), int(supercell[1]), 1)
    return slab, reduced_hkl


def prepare_existing_slab(input_slab, vacuum_total, supercell):
    slab = input_slab.copy()
    slab.set_pbc((True, True, True))
    # ASE's vacuum parameter is per side.  center() replaces the previous
    # c-direction empty space with the requested amount around the atoms.
    slab.center(vacuum=float(vacuum_total) / 2.0, axis=2)
    slab *= (int(supercell[0]), int(supercell[1]), 1)
    return slab


def apply_bottom_plane_constraint(slab, requested_planes, tolerance=0.20):
    planes = find_atomic_planes(slab, tolerance=tolerance)
    applied = min(int(requested_planes), max(len(planes) - 1, 0))
    indices = sorted(i for plane in planes[:applied] for i in plane)
    slab.set_constraint()
    if indices:
        slab.set_constraint(FixAtoms(indices=indices))
    return planes, applied, indices


def minimum_periodic_distance(atoms):
    _, _, distances = neighbor_list(
        "ijd", atoms, cutoff=natural_cutoffs(atoms, mult=2.0)
    )
    return float(np.min(distances)) if len(distances) else float("nan")


def severe_overlap_status(atoms):
    i_values, j_values, distances = neighbor_list(
        "ijd", atoms, cutoff=natural_cutoffs(atoms, mult=0.6)
    )
    mask = i_values != j_values
    overlap_distances = distances[mask]
    return {
        "has_overlap": bool(len(overlap_distances)),
        "count": int(len(overlap_distances)),
        "shortest": (
            float(np.min(overlap_distances))
            if len(overlap_distances)
            else None
        ),
    }


def validate_slab_geometry(slab, planes=None):
    planes = planes if planes is not None else find_atomic_planes(slab)
    gap = periodic_gap_analysis(slab)
    area = float(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1])))
    normal = surface_normal(slab)
    heights = np.asarray(slab.positions) @ normal
    thickness = float(np.ptp(heights)) if len(slab) else 0.0
    return {
        "atom_count": len(slab),
        "plane_count": len(planes),
        "atoms_per_plane": [len(plane) for plane in planes],
        "surface_area": area,
        "slab_thickness": thickness,
        "vacuum_gap": gap["largest_gap"],
        "minimum_distance": minimum_periodic_distance(slab),
        "overlap": severe_overlap_status(slab),
    }
