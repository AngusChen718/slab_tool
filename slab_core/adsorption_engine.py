"""Stage 2 orchestration for site search and adsorption candidates."""

from itertools import product

from .adsorbates import validate_adsorbate
from .adsorption import place_adsorbate
from .adsorption_sites import enumerate_adsorption_sites
from .adsorption_validation import validate_adsorption_candidate


def generate_adsorption_candidates(
    slab_candidate,
    adsorbate_spec,
    selected_site_ids=None,
    heights_A=(2.0,),
    tilts_deg=(0.0,),
    azimuths_deg=(0.0,),
    vacuum_total_A=15.0,
    max_candidates=120,
):
    adsorbate_report = validate_adsorbate(adsorbate_spec)
    if not adsorbate_report.passed:
        raise ValueError("吸附物未通過驗證：" + "；".join(adsorbate_report.errors))

    sites = enumerate_adsorption_sites(
        slab_candidate.atoms,
        metadata=slab_candidate.metadata,
        unique=True,
    )
    selected = set(selected_site_ids or [site.site_id for site in sites])
    chosen_sites = [site for site in sites if site.site_id in selected]
    if not chosen_sites:
        raise ValueError("沒有選擇有效的吸附位點")

    heights = tuple(float(value) for value in heights_A)
    tilts = tuple(float(value) for value in tilts_deg)
    azimuths = tuple(float(value) for value in azimuths_deg)
    if any(value <= 0 for value in heights):
        raise ValueError("所有吸附高度都必須大於 0 Å")
    if any(not 0 <= value <= 180 for value in tilts):
        raise ValueError("傾斜角必須介於 0° 與 180°")

    combinations = list(product(chosen_sites, heights, tilts, azimuths))
    if len(combinations) > int(max_candidates):
        raise ValueError(
            f"要求 {len(combinations)} 個候選，超過上限 {max_candidates}；"
            "請減少位點、高度或角度數量"
        )

    results = []
    for site, height, tilt, azimuth in combinations:
        candidate = place_adsorbate(
            slab_candidate=slab_candidate,
            adsorbate_spec=adsorbate_spec,
            site=site,
            height_A=height,
            tilt_deg=tilt,
            azimuth_deg=azimuth,
            vacuum_total_A=vacuum_total_A,
        )
        report = validate_adsorption_candidate(candidate)
        results.append((candidate, report))
    return adsorbate_report, sites, results
