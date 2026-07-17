from .analysis import assess_input, classify_bulk
from .builders import (
    apply_bottom_constraints,
    build_general_terminations,
    build_standard_surface,
    prepare_existing_slab,
)
from .validation import minimum_periodic_distance, validate_candidate


def generate_candidates(
    source_atoms,
    input_mode,
    hkl,
    atomic_layers,
    min_slab_size,
    vacuum_total,
    supercell,
    fixed_bottom_planes,
    allow_slab_as_bulk=False,
):
    assessment = assess_input(source_atoms)
    input_minimum = minimum_periodic_distance(source_atoms)

    if input_mode == "auto":
        if assessment["likely_slab"]:
            raise ValueError(
                f"輸入含約 {assessment['c_gap_A']:.2f} Å 週期空白，"
                "疑似既有 slab。請明確選擇『處理既有 Slab』；"
                "若要建立新晶面，請上傳 3D bulk。"
            )
        effective_mode = "bulk"
    elif input_mode == "bulk":
        if assessment["likely_slab"] and not allow_slab_as_bulk:
            raise ValueError(
                f"輸入含約 {assessment['c_gap_A']:.2f} Å 大空白，"
                "已阻止 slab 再次切割。"
            )
        effective_mode = "bulk"
    elif input_mode == "existing_slab":
        effective_mode = "existing_slab"
    else:
        raise ValueError(f"未知輸入模式：{input_mode}")

    if effective_mode == "existing_slab":
        route = {
            "builder_key": None,
            "lattice_type": "existing_slab",
            "hkl": None,
            "space_group_number": None,
            "space_group_symbol": None,
        }
        candidates = [
            prepare_existing_slab(source_atoms, vacuum_total, supercell)
        ]
    else:
        route = classify_bulk(source_atoms, hkl)
        if route["builder_key"]:
            candidates = [
                build_standard_surface(
                    route, atomic_layers, vacuum_total, supercell
                )
            ]
        else:
            candidates = build_general_terminations(
                route, min_slab_size, vacuum_total, supercell
            )

    results = []
    for candidate in candidates:
        planes = apply_bottom_constraints(candidate, fixed_bottom_planes)
        standard = candidate.metadata.get("route") == "standard"
        report = validate_candidate(
            candidate,
            requested_vacuum=vacuum_total,
            expected_atomic_layers=atomic_layers if standard else None,
            input_minimum_distance=input_minimum,
            strict_standard=standard,
        )
        candidate.metadata["top_composition"] = report.metrics.get(
            "top_composition", {}
        )
        candidate.metadata["bottom_composition"] = report.metrics.get(
            "bottom_composition", {}
        )
        results.append((candidate, report))
    return assessment, route, results
