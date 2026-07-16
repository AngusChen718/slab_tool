from functools import reduce
from math import gcd

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


STANDARD_BUILDERS = {
    ("fcc", (1, 1, 1)): "fcc111",
    ("fcc", (1, 0, 0)): "fcc100",
    ("fcc", (1, 1, 0)): "fcc110",
    ("bcc", (1, 1, 1)): "bcc111",
    ("bcc", (1, 0, 0)): "bcc100",
    ("bcc", (1, 1, 0)): "bcc110",
    ("hcp", (0, 0, 1)): "hcp0001",
}


def reduce_miller_index(hkl):
    values = tuple(int(v) for v in hkl)
    if values == (0, 0, 0):
        raise ValueError("Miller index 不可為 (0, 0, 0)")
    divisor = reduce(gcd, (abs(v) for v in values if v != 0))
    values = tuple(v // divisor for v in values)
    first_nonzero = next(v for v in values if v != 0)
    if first_nonzero < 0:
        values = tuple(-v for v in values)
    return values


def surface_normal(atoms):
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise ValueError("無法由晶胞 a、b 向量定義表面法向")
    normal = normal / norm
    if np.dot(normal, atoms.cell[2]) < 0:
        normal = -normal
    return normal


def normal_cell_height(atoms):
    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    if area <= 1e-12:
        raise ValueError("晶胞表面積必須大於零")
    return float(abs(atoms.get_volume()) / area)


def axis_gap_analysis(atoms, axis):
    if len(atoms) == 0:
        raise ValueError("結構沒有原子")
    scaled = np.mod(atoms.get_scaled_positions(wrap=False)[:, axis], 1.0)
    scaled.sort()
    cyclic = np.diff(np.concatenate([scaled, [scaled[0] + 1.0]]))
    fraction = float(np.max(cyclic))
    if axis == 2:
        physical_length = normal_cell_height(atoms)
    else:
        physical_length = float(np.linalg.norm(atoms.cell[axis]))
    return {
        "fraction": fraction,
        "gap_A": fraction * physical_length,
        "length_A": physical_length,
        "ratio": fraction,
    }


def assess_input(atoms, gap_threshold=3.5, ratio_threshold=0.25):
    c_gap = axis_gap_analysis(atoms, 2)
    likely_slab = (
        c_gap["gap_A"] >= gap_threshold
        and c_gap["ratio"] >= ratio_threshold
    )
    return {
        "formula": atoms.get_chemical_formula(),
        "atom_count": len(atoms),
        "c_gap_A": c_gap["gap_A"],
        "c_gap_ratio": c_gap["ratio"],
        "likely_slab": likely_slab,
    }


def find_atomic_planes(atoms, tolerance=0.20):
    normal = surface_normal(atoms)
    heights = np.asarray(atoms.positions) @ normal
    order = np.argsort(heights)
    planes = []
    for index in order:
        index = int(index)
        if not planes:
            planes.append([index])
            continue
        current_height = float(np.mean(heights[planes[-1]]))
        if abs(float(heights[index]) - current_height) <= tolerance:
            planes[-1].append(index)
        else:
            planes.append([index])
    return planes


def plane_compositions(atoms, planes):
    result = []
    symbols = atoms.get_chemical_symbols()
    for plane in planes:
        composition = {}
        for index in plane:
            symbol = symbols[index]
            composition[symbol] = composition.get(symbol, 0) + 1
        result.append(composition)
    return result


def classify_bulk(atoms, hkl, symprec=0.10):
    reduced_hkl = reduce_miller_index(hkl)
    structure = AseAtomsAdaptor.get_structure(atoms)
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=5)
    space_group = analyzer.get_space_group_number()
    conventional = analyzer.get_conventional_standard_structure()
    elements = sorted(str(el) for el in conventional.composition.elements)

    lattice_type = "general"
    if len(elements) == 1 and space_group == 225:
        lattice_type = "fcc"
    elif len(elements) == 1 and space_group == 229:
        lattice_type = "bcc"
    elif len(elements) == 1 and space_group == 194:
        lattice_type = "hcp"

    builder_key = STANDARD_BUILDERS.get((lattice_type, reduced_hkl))
    return {
        "space_group_number": space_group,
        "space_group_symbol": analyzer.get_space_group_symbol(),
        "lattice_type": lattice_type,
        "builder_key": builder_key,
        "hkl": reduced_hkl,
        "symbol": elements[0] if len(elements) == 1 else None,
        "a": float(conventional.lattice.a),
        "c": float(conventional.lattice.c),
        "conventional_structure": conventional,
    }
