"""Periodic two-dimensional adsorption-site detection."""

import numpy as np
from scipy.spatial import QhullError, Voronoi

from .analysis import find_atomic_planes, surface_normal
from .models import AdsorptionSite


def _surface_basis(atoms):
    a = np.asarray(atoms.cell[0], dtype=float)
    b = np.asarray(atoms.cell[1], dtype=float)
    e1 = a / np.linalg.norm(a)
    b_perpendicular = b - np.dot(b, e1) * e1
    e2 = b_perpendicular / np.linalg.norm(b_perpendicular)
    matrix = np.array([
        [np.dot(a, e1), np.dot(b, e1)],
        [np.dot(a, e2), np.dot(b, e2)],
    ])
    return a, b, e1, e2, matrix


def _periodic_lateral_distance(first, second, a, b):
    best_distance = float("inf")
    best_delta = None
    best_shift = None
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            delta_fractional = second + np.array([i, j], dtype=float) - first
            delta = delta_fractional[0] * a + delta_fractional[1] * b
            distance = float(np.linalg.norm(delta))
            if distance < best_distance:
                best_distance = distance
                best_delta = delta_fractional
                best_shift = (i, j)
    return best_distance, best_delta, best_shift


def _fractional_key(value, digits=7):
    wrapped = np.mod(np.asarray(value, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=10 ** (-digits))] = 0.0
    return tuple(float(v) for v in np.round(wrapped, digits))


def _nearest_surface_spacing(fractional, a, b):
    nearest = float("inf")
    for i, first in enumerate(fractional):
        for j, second in enumerate(fractional):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    if i == j and sx == 0 and sy == 0:
                        continue
                    delta_fractional = (
                        second + np.array([sx, sy], dtype=float) - first
                    )
                    distance = float(np.linalg.norm(
                        delta_fractional[0] * a + delta_fractional[1] * b
                    ))
                    if distance > 1e-8:
                        nearest = min(nearest, distance)
    if not np.isfinite(nearest):
        raise ValueError("無法判定表面最近鄰距離")
    return nearest


def _cartesian_site(atoms, fractional_xy, surface_height):
    normal = surface_normal(atoms)
    value = (
        float(fractional_xy[0]) * np.asarray(atoms.cell[0])
        + float(fractional_xy[1]) * np.asarray(atoms.cell[1])
        + float(surface_height) * normal
    )
    return tuple(float(v) for v in value)


def _registry_type(atoms, fractional_xy, planes, nearest_spacing, metadata):
    if not (
        metadata.get("lattice_type") == "fcc"
        and tuple(metadata.get("hkl") or ()) == (1, 1, 1)
        and len(planes) >= 3
    ):
        return "hollow_3"
    a = np.asarray(atoms.cell[0])
    b = np.asarray(atoms.cell[1])
    scaled = np.mod(atoms.get_scaled_positions(wrap=False)[:, :2], 1.0)

    def plane_minimum(plane):
        return min(
            _periodic_lateral_distance(fractional_xy, scaled[index], a, b)[0]
            for index in plane
        )

    tolerance = max(0.12, 0.08 * nearest_spacing)
    if plane_minimum(planes[-2]) <= tolerance:
        return "hcp_hollow"
    if plane_minimum(planes[-3]) <= tolerance:
        return "fcc_hollow"
    return "hollow_3"


def _site_signature(site_type, symbols, distances):
    return (
        site_type,
        tuple(sorted(symbols)),
        tuple(sorted(round(float(value), 3) for value in distances)),
    )


def enumerate_adsorption_sites(atoms, metadata=None, unique=True):
    """Return top, nearest-neighbor bridge, and Voronoi hollow sites.

    Coordinates are evaluated with two-dimensional periodic boundary
    conditions.  When ``unique`` is true, equivalent local-environment
    signatures are collapsed and their multiplicity is retained.
    """
    metadata = dict(metadata or {})
    planes = find_atomic_planes(atoms)
    if not planes:
        raise ValueError("Slab 沒有可辨識的原子平面")
    top_plane = planes[-1]
    normal = surface_normal(atoms)
    heights = np.asarray(atoms.positions) @ normal
    surface_height = float(np.mean(heights[top_plane]))
    scaled = np.mod(atoms.get_scaled_positions(wrap=False), 1.0)
    top_fractional = scaled[top_plane, :2]
    a, b, _, _, transform = _surface_basis(atoms)
    nearest_spacing = _nearest_surface_spacing(top_fractional, a, b)
    symbols = atoms.get_chemical_symbols()
    raw_sites = []

    # Top sites.
    seen_top = set()
    for atom_index, fractional_xy in zip(top_plane, top_fractional):
        key = _fractional_key(fractional_xy)
        if key in seen_top:
            continue
        seen_top.add(key)
        raw_sites.append({
            "site_type": "top",
            "fractional_xy": key,
            "coordination": 1,
            "surface_indices": [int(atom_index)],
            "symbols": [symbols[atom_index]],
            "distances": [0.0],
        })

    # Nearest-neighbor bridges, including bonds to a periodic image of the
    # same atom when the primitive surface cell contains one top atom.
    seen_bridge = set()
    for local_i, first in enumerate(top_fractional):
        for local_j, second in enumerate(top_fractional):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    if local_i == local_j and sx == 0 and sy == 0:
                        continue
                    delta_fractional = (
                        second + np.array([sx, sy], dtype=float) - first
                    )
                    distance = float(np.linalg.norm(
                        delta_fractional[0] * a + delta_fractional[1] * b
                    ))
                    if not (0.90 * nearest_spacing <= distance <= 1.10 * nearest_spacing):
                        continue
                    midpoint = first + 0.5 * delta_fractional
                    key = _fractional_key(midpoint)
                    if key in seen_bridge:
                        continue
                    seen_bridge.add(key)
                    indices = [int(top_plane[local_i]), int(top_plane[local_j])]
                    raw_sites.append({
                        "site_type": "bridge",
                        "fractional_xy": key,
                        "coordination": 2,
                        "surface_indices": indices,
                        "symbols": [symbols[index] for index in indices],
                        "distances": [0.5 * distance, 0.5 * distance],
                    })

    # Voronoi vertices are points equidistant from three or more periodic
    # surface atoms: threefold hollows on triangular lattices and fourfold
    # hollows on square lattices.
    replicated_fractional = []
    replicated_indices = []
    for sx in (-2, -1, 0, 1, 2):
        for sy in (-2, -1, 0, 1, 2):
            for local_index, value in enumerate(top_fractional):
                replicated_fractional.append(value + [sx, sy])
                replicated_indices.append(int(top_plane[local_index]))
    replicated_fractional = np.asarray(replicated_fractional)
    points_2d = (transform @ replicated_fractional.T).T
    try:
        voronoi = Voronoi(points_2d)
    except QhullError as exc:
        raise ValueError(f"表面位點 Voronoi 分析失敗：{exc}") from exc

    seen_hollow = set()
    inverse_transform = np.linalg.inv(transform)
    for vertex in voronoi.vertices:
        fractional_xy = inverse_transform @ np.asarray(vertex)
        # Ignore vertices near the outer hull of the replicated cloud.  Their
        # Voronoi cells are incomplete and mapping them back with modulo can
        # create spurious threefold sites in an otherwise fourfold lattice.
        if np.any(fractional_xy < -1e-8) or np.any(fractional_xy >= 1.0 - 1e-8):
            continue
        key = _fractional_key(fractional_xy)
        if key in seen_hollow:
            continue
        distances = np.linalg.norm(points_2d - vertex, axis=1)
        minimum = float(np.min(distances))
        neighbor_rows = np.where(
            np.abs(distances - minimum) <= max(1e-5, 0.015 * nearest_spacing)
        )[0]
        coordination = int(len(neighbor_rows))
        if coordination < 3 or coordination > 6:
            continue
        if not (0.45 * nearest_spacing <= minimum <= 0.90 * nearest_spacing):
            continue
        if coordination == 4:
            site_type = (
                "fourfold_hollow"
                if metadata.get("lattice_type") == "fcc"
                and tuple(metadata.get("hkl") or ()) == (1, 0, 0)
                else "hollow_4"
            )
        else:
            site_type = _registry_type(
                atoms, np.asarray(key), planes, nearest_spacing, metadata
            )
        indices = [replicated_indices[row] for row in neighbor_rows]
        seen_hollow.add(key)
        raw_sites.append({
            "site_type": site_type,
            "fractional_xy": key,
            "coordination": coordination,
            "surface_indices": indices,
            "symbols": [symbols[index] for index in indices],
            "distances": [minimum] * coordination,
        })

    if not raw_sites:
        raise ValueError("沒有找到任何吸附位點")

    if unique:
        grouped = {}
        for site in raw_sites:
            signature = _site_signature(
                site["site_type"], site["symbols"], site["distances"]
            )
            score = min(
                site["fractional_xy"][0],
                1.0 - site["fractional_xy"][0],
                site["fractional_xy"][1],
                1.0 - site["fractional_xy"][1],
            )
            if signature not in grouped:
                grouped[signature] = {**site, "multiplicity": 1, "score": score}
            else:
                grouped[signature]["multiplicity"] += 1
                if score > grouped[signature]["score"]:
                    multiplicity = grouped[signature]["multiplicity"]
                    grouped[signature] = {
                        **site,
                        "multiplicity": multiplicity,
                        "score": score,
                    }
        selected_sites = list(grouped.values())
    else:
        selected_sites = [{**site, "multiplicity": 1} for site in raw_sites]

    order = {
        "top": 0,
        "bridge": 1,
        "fcc_hollow": 2,
        "hcp_hollow": 3,
        "hollow_3": 4,
        "hollow_4": 5,
        "fourfold_hollow": 6,
    }
    selected_sites.sort(key=lambda site: (order.get(site["site_type"], 99), site["fractional_xy"]))
    counters = {}
    result = []
    for site in selected_sites:
        site_type = site["site_type"]
        counters[site_type] = counters.get(site_type, 0) + 1
        prefix = {
            "top": "T",
            "bridge": "B",
            "fcc_hollow": "F",
            "hcp_hollow": "H",
            "fourfold_hollow": "Q",
            "hollow_3": "R",
            "hollow_4": "X",
        }.get(site_type, "X")
        fractional_xy = tuple(float(v) for v in site["fractional_xy"])
        result.append(AdsorptionSite(
            site_id=f"{prefix}{counters[site_type]}",
            site_type=site_type,
            fractional_xy=fractional_xy,
            cartesian=_cartesian_site(atoms, fractional_xy, surface_height),
            coordination=int(site["coordination"]),
            surface_indices=[int(v) for v in site["surface_indices"]],
            multiplicity=int(site.get("multiplicity", 1)),
            metadata={
                "surface_height_A": surface_height,
                "surface_nearest_neighbor_A": nearest_spacing,
                "coordinating_symbols": site["symbols"],
            },
        ))
    return result
