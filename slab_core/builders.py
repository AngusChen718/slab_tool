import numpy as np
from ase.build import (
    bcc100,
    bcc110,
    bcc111,
    fcc100,
    fcc110,
    fcc111,
    hcp0001,
)
from ase.constraints import FixAtoms
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

from .analysis import find_atomic_planes, plane_compositions
from .models import BuildCandidate


SPECIALIZED_BUILDERS = {
    "fcc111": fcc111,
    "fcc100": fcc100,
    "fcc110": fcc110,
    "bcc111": bcc111,
    "bcc100": bcc100,
    "bcc110": bcc110,
    "hcp0001": hcp0001,
}


def build_standard_surface(route, atomic_layers, vacuum_total, supercell):
    key = route["builder_key"]
    if key not in SPECIALIZED_BUILDERS:
        raise ValueError("此結構／晶面沒有對應的標準 ASE builder")
    kwargs = {
        "symbol": route["symbol"],
        "size": (int(supercell[0]), int(supercell[1]), int(atomic_layers)),
        "a": route["a"],
        "vacuum": float(vacuum_total) / 2.0,
    }
    if key == "hcp0001":
        kwargs["c"] = route["c"]
    slab = SPECIALIZED_BUILDERS[key](**kwargs)
    slab.set_pbc((True, True, True))
    return BuildCandidate(
        atoms=slab,
        candidate_id=f"standard_{key}",
        builder=f"ase.build.{key}",
        metadata={
            "route": "standard",
            "hkl": list(route["hkl"]),
            "requested_atomic_layers": int(atomic_layers),
            "space_group_number": route["space_group_number"],
            "space_group_symbol": route["space_group_symbol"],
            "lattice_type": route["lattice_type"],
        },
    )


def _termination_metadata(pmg_slab, ase_slab):
    planes = find_atomic_planes(ase_slab)
    compositions = plane_compositions(ase_slab, planes)
    try:
        symmetric = bool(pmg_slab.is_symmetric())
    except Exception:
        symmetric = False
    try:
        polar = bool(pmg_slab.is_polar())
    except Exception:
        polar = None
    return {
        "shift": float(pmg_slab.shift),
        "symmetric": symmetric,
        "polar": polar,
        "top_composition": compositions[-1] if compositions else {},
        "bottom_composition": compositions[0] if compositions else {},
        "slab_formula": ase_slab.get_chemical_formula(),
    }


def build_general_terminations(
    route,
    min_slab_size,
    vacuum_total,
    supercell,
    max_normal_search=20,
):
    generator = SlabGenerator(
        initial_structure=route["conventional_structure"],
        miller_index=route["hkl"],
        min_slab_size=float(min_slab_size),
        min_vacuum_size=float(vacuum_total),
        lll_reduce=False,
        center_slab=True,
        in_unit_planes=False,
        primitive=True,
        max_normal_search=int(max_normal_search),
        reorient_lattice=True,
    )
    raw_slabs = [
        (slab, False) for slab in generator.get_slabs(symmetrize=False)
    ]
    raw_slabs.extend(
        (slab, True) for slab in generator.get_slabs(symmetrize=True)
    )
    if not raw_slabs:
        raise ValueError("pymatgen 未產生任何 termination 候選")

    # Keep stoichiometric asymmetric slabs and symmetric alternatives, while
    # removing exact structural duplicates returned by both calls.
    matcher = StructureMatcher(
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )
    slabs = []
    for slab, symmetrized in raw_slabs:
        duplicate = any(matcher.fit(existing[0], slab) for existing in slabs)
        if not duplicate:
            slabs.append((slab, symmetrized))

    candidates = []
    for index, (pmg_slab, symmetrized) in enumerate(slabs):
        ase_slab = AseAtomsAdaptor.get_atoms(pmg_slab)
        ase_slab.set_pbc((True, True, True))
        # Normalize to the requested total normal vacuum, then repeat only in-plane.
        ase_slab.center(vacuum=float(vacuum_total) / 2.0, axis=2)
        ase_slab *= (int(supercell[0]), int(supercell[1]), 1)
        metadata = {
            "route": "general",
            "hkl": list(route["hkl"]),
            "space_group_number": route["space_group_number"],
            "space_group_symbol": route["space_group_symbol"],
            "termination_index": index,
            "symmetrized_by_atom_removal": bool(symmetrized),
            **_termination_metadata(pmg_slab, ase_slab),
        }
        candidates.append(
            BuildCandidate(
                atoms=ase_slab,
                candidate_id=f"termination_{index:02d}",
                builder="pymatgen.core.surface.SlabGenerator",
                metadata=metadata,
            )
        )
    return candidates


def prepare_existing_slab(atoms, vacuum_total, supercell):
    slab = atoms.copy()
    slab.set_pbc((True, True, True))
    slab.center(vacuum=float(vacuum_total) / 2.0, axis=2)
    slab *= (int(supercell[0]), int(supercell[1]), 1)
    return BuildCandidate(
        atoms=slab,
        candidate_id="existing_slab",
        builder="existing-slab/no-recut",
        metadata={"route": "existing_slab", "hkl": None},
    )


def apply_bottom_constraints(candidate, requested_planes, tolerance=0.20):
    slab = candidate.atoms
    planes = find_atomic_planes(slab, tolerance=tolerance)
    applied = min(int(requested_planes), max(len(planes) - 1, 0))
    fixed_indices = sorted(
        index for plane in planes[:applied] for index in plane
    )
    slab.set_constraint()
    if fixed_indices:
        slab.set_constraint(FixAtoms(indices=fixed_indices))
    candidate.metadata.update({
        "fixed_plane_count": applied,
        "fixed_atom_count": len(fixed_indices),
        "fixed_indices": fixed_indices,
    })
    return planes
