"""Adsorbate construction and molecule-level geometry validation."""

from collections import deque

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.data import covalent_radii

from .models import AdsorbateSpec, ValidationReport


BUILTIN_ADSORBATES = {
    "H": {"anchor": 0, "multiplicity": 2},
    "O": {"anchor": 0, "multiplicity": 3},
    "N": {"anchor": 0, "multiplicity": 4},
    "C": {"anchor": 0, "multiplicity": 3},
    "H2": {"anchor": 0, "multiplicity": 1},
    "O2": {"anchor": 0, "multiplicity": 3},
    "N2": {"anchor": 0, "multiplicity": 1},
    "CO": {"anchor": 1, "multiplicity": 1},  # ASE order is O, C: C-down.
    "OH": {"anchor": 0, "multiplicity": 2},
    "H2O": {"anchor": 0, "multiplicity": 1},
    "NH3": {"anchor": 0, "multiplicity": 1},
}


def _builtin_atoms(name):
    if name in {"H", "O", "N", "C"}:
        return Atoms(name, positions=[[0.0, 0.0, 0.0]])
    if name == "OH":
        return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    return molecule(name)


def make_adsorbate(
    name,
    anchor_index=None,
    charge=0,
    spin_multiplicity=None,
):
    if name not in BUILTIN_ADSORBATES:
        raise ValueError(f"未知的內建吸附物：{name}")
    defaults = BUILTIN_ADSORBATES[name]
    atoms = _builtin_atoms(name)
    anchor = defaults["anchor"] if anchor_index is None else int(anchor_index)
    multiplicity = (
        defaults["multiplicity"]
        if spin_multiplicity is None
        else int(spin_multiplicity)
    )
    return AdsorbateSpec(
        name=name,
        atoms=atoms,
        anchor_index=anchor,
        charge=int(charge),
        spin_multiplicity=multiplicity,
        metadata={"source": "ASE/built-in"},
    )


def make_uploaded_adsorbate(
    atoms,
    name="uploaded_adsorbate",
    anchor_index=0,
    charge=0,
    spin_multiplicity=None,
):
    copied = atoms.copy()
    copied.set_pbc(False)
    copied.set_cell([0.0, 0.0, 0.0])
    return AdsorbateSpec(
        name=str(name),
        atoms=copied,
        anchor_index=int(anchor_index),
        charge=int(charge),
        spin_multiplicity=(
            None if spin_multiplicity is None else int(spin_multiplicity)
        ),
        metadata={"source": "uploaded"},
    )


def _pair_geometry(atoms):
    positions = np.asarray(atoms.positions, dtype=float)
    numbers = atoms.numbers
    bonds = []
    severe = []
    distances = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = float(np.linalg.norm(positions[j] - positions[i]))
            radii_sum = float(covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
            distances.append(distance)
            if distance < max(0.25, 0.45 * radii_sum):
                severe.append((i, j, distance))
            if distance <= 1.30 * radii_sum:
                bonds.append((i, j, distance))
    return bonds, severe, distances


def _is_connected(atom_count, bonds):
    if atom_count <= 1:
        return True
    graph = {index: [] for index in range(atom_count)}
    for i, j, _ in bonds:
        graph[i].append(j)
        graph[j].append(i)
    visited = {0}
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == atom_count


def validate_adsorbate(spec):
    atoms = spec.atoms
    errors = []
    warnings = []
    checks = {}
    if len(atoms) == 0:
        return ValidationReport(status="failed", errors=["吸附物沒有原子"])

    valid_anchor = 0 <= int(spec.anchor_index) < len(atoms)
    checks["valid_anchor"] = valid_anchor
    if not valid_anchor:
        errors.append("錨定原子索引超出範圍")

    finite = bool(np.all(np.isfinite(atoms.positions)))
    checks["finite_coordinates"] = finite
    if not finite:
        errors.append("吸附物包含非有限座標")

    bonds, severe, distances = _pair_geometry(atoms)
    checks["no_internal_overlap"] = not severe
    if severe:
        errors.append(
            f"吸附物內有 {len(severe)} 組原子嚴重重疊；"
            f"最短 {min(item[2] for item in severe):.4f} Å"
        )

    connected = _is_connected(len(atoms), bonds)
    checks["molecule_connected"] = connected
    if len(atoms) > 1 and not connected:
        errors.append("吸附物不是單一連通分子；請檢查鍵長或上傳內容")

    if spec.spin_multiplicity is None:
        warnings.append("未提供自旋多重度；產生 VASP 設定前必須人工確認")
    elif spec.spin_multiplicity < 1:
        errors.append("自旋多重度必須至少為 1")

    metrics = {
        "name": spec.name,
        "formula": atoms.get_chemical_formula(),
        "atom_count": len(atoms),
        "anchor_index": int(spec.anchor_index),
        "anchor_symbol": (
            atoms[int(spec.anchor_index)].symbol if valid_anchor else None
        ),
        "charge": int(spec.charge),
        "spin_multiplicity": spec.spin_multiplicity,
        "bond_count": len(bonds),
        "bonds": [
            {"i": i, "j": j, "distance_A": distance}
            for i, j, distance in bonds
        ],
        "minimum_internal_distance_A": (
            min(distances) if distances else None
        ),
    }
    return ValidationReport(
        status="passed" if not errors else "failed",
        errors=errors,
        warnings=warnings,
        metrics=metrics,
        checks=checks,
    )
