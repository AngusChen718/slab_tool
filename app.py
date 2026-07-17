import datetime
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import py3Dmol
import streamlit as st
from ase.io import read, write

from slab_core import (
    BUILTIN_ADSORBATES,
    assess_input,
    classify_bulk,
    enumerate_adsorption_sites,
    generate_adsorption_candidates,
    generate_candidates,
    make_adsorbate,
    make_uploaded_adsorbate,
    validate_adsorbate,
)


APP_VERSION = "2026-07-17-stage2.3-v1"


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "slab"


def atoms_to_poscar(atoms):
    buffer = io.StringIO()
    write(buffer, atoms, format="vasp", vasp5=True, direct=False)
    return buffer.getvalue()


def atoms_to_xyz(atoms):
    buffer = io.StringIO()
    write(buffer, atoms, format="xyz")
    return buffer.getvalue()


def parse_number_list(text, label, minimum=None, maximum=None):
    try:
        values = tuple(
            float(part.strip())
            for part in str(text).replace("；", ",").split(",")
            if part.strip()
        )
    except ValueError as exc:
        raise ValueError(f"{label} 必須是以逗號分隔的數字") from exc
    if not values:
        raise ValueError(f"{label} 至少需要一個數值")
    if minimum is not None and any(value < minimum for value in values):
        raise ValueError(f"{label} 不可小於 {minimum}")
    if maximum is not None and any(value > maximum for value in values):
        raise ValueError(f"{label} 不可大於 {maximum}")
    return values


def show_mapping_table(mapping):
    def display_value(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=json_default)
        return str(value)

    st.table([
        {"項目": str(key), "內容": display_value(value)}
        for key, value in mapping.items()
    ])


def read_adsorbate_upload(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".xyz"):
        file_format = "xyz"
    elif name.endswith(".cif"):
        file_format = "cif"
    else:
        file_format = "vasp"
    payload = uploaded_file.getvalue().decode("utf-8")
    atoms = read(io.StringIO(payload), format=file_format)
    atoms.set_pbc(False)
    return atoms


def make_package(input_text, candidate, report, settings, input_name):
    poscar = atoms_to_poscar(candidate.atoms)
    report_dict = report.to_dict()
    report_json = json.dumps(
        report_dict, ensure_ascii=False, indent=2, default=json_default
    )
    settings_json = json.dumps(
        settings, ensure_ascii=False, indent=2, default=json_default
    )
    log = (
        "Slab Builder Stage 2.3\n"
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"Input: {input_name}\n"
        f"Builder: {candidate.builder}\n"
        f"Candidate: {candidate.candidate_id}\n"
        f"Validation: {report.status}\n"
        "This is an unrelaxed initial structure. DFT convergence is not certified.\n"
    )
    memory = io.BytesIO()
    folder = safe_name(f"{input_name}_{candidate.candidate_id}") + "/"
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(folder + "input_POSCAR", input_text)
        archive.writestr(folder + "POSCAR", poscar)
        archive.writestr(folder + "validation_report.json", report_json)
        archive.writestr(folder + "generation_settings.json", settings_json)
        archive.writestr(folder + "README_or_log.txt", log)
    return memory.getvalue(), poscar, report_json, settings_json


def make_adsorption_package(base_candidate, adsorbate_spec, candidate, report):
    poscar = atoms_to_poscar(candidate.atoms)
    report_json = json.dumps(
        report.to_dict(), ensure_ascii=False, indent=2, default=json_default
    )
    site_json = json.dumps(
        candidate.metadata.get("site", {}),
        ensure_ascii=False,
        indent=2,
        default=json_default,
    )
    settings = {
        key: candidate.metadata.get(key)
        for key in (
            "base_candidate_id", "adsorbate_name", "adsorbate_formula",
            "anchor_local_index", "requested_height_A", "tilt_deg",
            "azimuth_deg", "requested_vacuum_A",
        )
    }
    settings_json = json.dumps(
        settings, ensure_ascii=False, indent=2, default=json_default
    )
    log = (
        "Slab Builder Stage 2.3 adsorption candidate\n"
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"Candidate: {candidate.candidate_id}\n"
        f"Validation: {report.status}\n"
        "Initial geometry only. DFT relaxation and convergence are required.\n"
    )
    memory = io.BytesIO()
    folder = safe_name(candidate.candidate_id) + "/"
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(folder + "input_slab_POSCAR", atoms_to_poscar(base_candidate.atoms))
        archive.writestr(folder + "adsorbate.xyz", atoms_to_xyz(adsorbate_spec.atoms))
        archive.writestr(folder + "POSCAR", poscar)
        archive.writestr(folder + "site.json", site_json)
        archive.writestr(folder + "generation_settings.json", settings_json)
        archive.writestr(folder + "validation_report.json", report_json)
        archive.writestr(folder + "README_or_log.txt", log)
    return memory.getvalue(), poscar


def draw_cell(view, cell, shift):
    a, b, c = np.asarray(cell, dtype=float)
    origin = np.asarray(shift, dtype=float)
    corners = [
        origin,
        origin + a,
        origin + b,
        origin + c,
        origin + a + b,
        origin + a + c,
        origin + b + c,
        origin + a + b + c,
    ]
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]
    for start_index, end_index in edges:
        start, end = corners[start_index], corners[end_index]
        view.addLine({
            "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
            "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
            "color": "#555555",
            "linewidth": 1.4,
        })


def render_atoms(atoms, widget_key, height=470, wrap_positions=True):
    controls = st.columns(3)
    show_bonds = controls[0].checkbox("顯示鍵結", True, key=f"bonds_{widget_key}")
    show_cell = controls[1].checkbox("顯示晶胞", True, key=f"cell_{widget_key}")
    orientation = controls[2].selectbox(
        "觀看方向", ["側視", "俯視", "立體"], key=f"view_{widget_key}"
    )

    display_atoms = atoms.copy()
    if wrap_positions:
        display_atoms.wrap(eps=1e-10)
    shift = -0.5 * np.sum(np.asarray(display_atoms.cell), axis=0)
    display_atoms.translate(shift)
    display_atoms.set_pbc(False)
    xyz_buffer = io.StringIO()
    write(xyz_buffer, display_atoms, format="xyz")

    view = py3Dmol.view(width=760, height=height)
    view.addModel(xyz_buffer.getvalue(), "xyz")
    style = {"sphere": {"colorscheme": "Jmol", "scale": 0.31}}
    if show_bonds:
        style["stick"] = {"colorscheme": "Jmol", "radius": 0.09}
    view.setStyle(style)
    if show_cell:
        draw_cell(view, atoms.cell, shift)
    view.setBackgroundColor("#F8F9FA")
    view.zoomTo()
    view.setView([0, 0, 0, 0, 0, 0, 0, 1])
    if orientation == "側視":
        view.rotate(90, "x")
    elif orientation == "立體":
        view.rotate(-55, "x")
        view.rotate(30, "z")
    view.zoomTo()
    # Embed py3Dmol directly.  This avoids stmol's legacy Jupyter dependency
    # chain, which can fail on a clean Streamlit deployment.
    st.iframe(view._make_html(), height=height, width=780)


def report_table(report):
    metrics = report.metrics
    rows = {
        "驗證狀態": report.status,
        "Builder": metrics.get("builder"),
        "原子數": metrics.get("atom_count"),
        "實際原子平面": metrics.get("atomic_plane_count"),
        "每層原子數": metrics.get("atoms_per_plane"),
        "Slab 厚度": f"{metrics.get('slab_thickness_A', 0):.4f} Å",
        "實測總真空": f"{metrics.get('vacuum_A', 0):.4f} Å",
        "最短週期距離": f"{metrics.get('minimum_distance_A', 0):.4f} Å",
        "表面積": f"{metrics.get('surface_area_A2', 0):.4f} Å²",
        "底面組成": metrics.get("bottom_composition"),
        "頂面組成": metrics.get("top_composition"),
        "固定平面／原子": (
            f"{metrics.get('fixed_plane_count', 0)} / "
            f"{metrics.get('fixed_atom_count', 0)}"
        ),
    }
    show_mapping_table(rows)
    for error in report.errors:
        st.error(error)
    for warning in report.warnings:
        st.warning(warning)


def adsorption_report_table(report):
    metrics = report.metrics
    rows = {
        "驗證狀態": report.status,
        "吸附物": metrics.get("adsorbate_formula"),
        "位點": f"{metrics.get('site_id')} / {metrics.get('site_type')}",
        "設定／實測高度": (
            f"{metrics.get('requested_height_A', 0):.4f} / "
            f"{metrics.get('measured_anchor_height_A', 0):.4f} Å"
        ),
        "吸附物－表面最短距離": (
            f"{metrics.get('minimum_adsorbate_surface_distance_A', 0):.4f} Å"
        ),
        "吸附後週期真空": f"{metrics.get('remaining_vacuum_A', 0):.4f} Å",
        "吸附物週期影像距離": (
            f"{metrics.get('adsorbate_image_distance_A', 0):.4f} Å"
        ),
        "近似覆蓋率": metrics.get("coverage_ML_approx"),
        "傾斜角／方位角": (
            f"{metrics.get('tilt_deg')}° / {metrics.get('azimuth_deg')}°"
        ),
    }
    show_mapping_table(rows)
    for error in report.errors:
        st.error(error)
    for warning in report.warnings:
        st.warning(warning)


st.set_page_config(page_title="Slab Builder Stage 2.3", layout="wide")
st.title("Slab Builder — Surface & Adsorption Validation")
st.caption(f"版本：{APP_VERSION}｜驗證失敗的候選不可下載")

if "workspaces" not in st.session_state:
    st.session_state.workspaces = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "adsorption_pending" not in st.session_state:
    st.session_state.adsorption_pending = None

st.sidebar.header("參數設定")
mode_label = st.sidebar.radio(
    "輸入處理模式",
    ["自動判斷", "由 3D Bulk 建立新 Slab", "處理既有 Slab"],
)
mode_map = {
    "自動判斷": "auto",
    "由 3D Bulk 建立新 Slab": "bulk",
    "處理既有 Slab": "existing_slab",
}
input_mode = mode_map[mode_label]

h = k = l = 0
if input_mode != "existing_slab":
    st.sidebar.markdown("**Miller index**")
    cols = st.sidebar.columns(3)
    h = cols[0].number_input("h", value=1, step=1)
    k = cols[1].number_input("k", value=1, step=1)
    l = cols[2].number_input("l", value=1, step=1)

atomic_layers = st.sidebar.number_input(
    "標準金屬：實際原子層數", min_value=2, value=4, step=1
)
min_slab_size = st.sidebar.number_input(
    "一般材料：最小 Slab 厚度 (Å)", min_value=3.0, value=10.0, step=1.0
)
vacuum_total = st.sidebar.number_input(
    "週期影像間總真空 (Å)", min_value=5.0, value=15.0, step=1.0
)
super_cols = st.sidebar.columns(2)
super_x = super_cols[0].number_input("X 倍數", min_value=1, value=3)
super_y = super_cols[1].number_input("Y 倍數", min_value=1, value=3)
fixed_planes = st.sidebar.number_input(
    "固定底部完整平面", min_value=0, value=2, step=1
)
allow_override = False
if input_mode == "bulk":
    allow_override = st.sidebar.checkbox("進階：忽略疑似 Slab 警告", False)

uploaded = st.file_uploader("上傳 POSCAR", type=None, accept_multiple_files=False)

source_atoms = None
source_text = None
route_preview = None
request_signature = None
if uploaded is not None:
    try:
        source_text = uploaded.getvalue().decode("utf-8")
        source_atoms = read(io.StringIO(source_text), format="vasp")
        signature_payload = {
            "source": source_text,
            "input_mode": input_mode,
            "hkl": [h, k, l],
            "atomic_layers": atomic_layers,
            "min_slab_size": min_slab_size,
            "vacuum_total": vacuum_total,
            "supercell": [super_x, super_y],
            "fixed_planes": fixed_planes,
            "allow_override": allow_override,
        }
        request_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if (
            st.session_state.pending is not None
            and st.session_state.pending.get("request_signature")
            != request_signature
        ):
            st.session_state.pending = None
            st.session_state.adsorption_pending = None
        assessment = assess_input(source_atoms)
        preview = {
            "化學式": assessment["formula"],
            "原子數": assessment["atom_count"],
            "最大 c 向週期空白": f"{assessment['c_gap_A']:.4f} Å",
            "輸入判定": "疑似既有 Slab" if assessment["likely_slab"] else "3D Bulk 候選",
        }
        if input_mode != "existing_slab":
            route_preview = classify_bulk(source_atoms, (h, k, l))
            preview.update({
                "空間群": (
                    f"{route_preview['space_group_symbol']} "
                    f"({route_preview['space_group_number']})"
                ),
                "建構路由": (
                    f"ASE {route_preview['builder_key']}"
                    if route_preview["builder_key"]
                    else "pymatgen termination workflow"
                ),
            })
        st.subheader("輸入預檢")
        show_mapping_table(preview)
    except Exception as exc:
        st.error(f"輸入讀取／分析失敗：{exc}")

if source_atoms is not None and st.button("產生並驗證候選", type="primary"):
    try:
        assessment, route, results = generate_candidates(
            source_atoms=source_atoms,
            input_mode=input_mode,
            hkl=(h, k, l),
            atomic_layers=atomic_layers,
            min_slab_size=min_slab_size,
            vacuum_total=vacuum_total,
            supercell=(super_x, super_y),
            fixed_bottom_planes=fixed_planes,
            allow_slab_as_bulk=allow_override,
        )
        settings = {
            "app_version": APP_VERSION,
            "input_mode": input_mode,
            "hkl": [h, k, l] if input_mode != "existing_slab" else None,
            "atomic_layers": atomic_layers,
            "min_slab_size_A": min_slab_size,
            "vacuum_total_A": vacuum_total,
            "supercell": [super_x, super_y, 1],
            "fixed_bottom_planes": fixed_planes,
            "input_assessment": assessment,
            "route": {key: value for key, value in route.items() if key != "conventional_structure"},
        }
        st.session_state.pending = {
            "input_name": uploaded.name,
            "input_text": source_text,
            "results": results,
            "settings": settings,
            "request_signature": request_signature,
        }
        st.session_state.adsorption_pending = None
        st.success(f"已產生 {len(results)} 個候選")
    except Exception as exc:
        st.session_state.pending = None
        st.error(str(exc))

if uploaded is None:
    st.session_state.pending = None
    st.session_state.adsorption_pending = None
pending = st.session_state.pending
if pending:
    st.header("候選結構")
    labels = []
    for candidate, report in pending["results"]:
        top = report.metrics.get("top_composition", {})
        bottom = report.metrics.get("bottom_composition", {})
        labels.append(
            f"{candidate.candidate_id}｜{report.status}｜bottom {bottom}｜top {top}"
        )
    selected = st.selectbox("選擇 termination／候選", range(len(labels)), format_func=lambda i: labels[i])
    candidate, report = pending["results"][selected]
    render_atoms(candidate.atoms, f"pending_{selected}")
    report_table(report)

    if report.passed:
        package, poscar, report_json, settings_json = make_package(
            pending["input_text"], candidate, report,
            pending["settings"], pending["input_name"],
        )
        columns = st.columns(2)
        columns[0].download_button(
            "下載已驗證結構包",
            data=package,
            file_name=safe_name(f"{pending['input_name']}_{candidate.candidate_id}.zip"),
            mime="application/zip",
        )
        if columns[1].button("加入工作區"):
            st.session_state.workspaces.append({
                "name": f"{pending['input_name']} — {candidate.candidate_id}",
                "poscar": poscar,
                "package": package,
                "report": report.to_dict(),
                "filename": safe_name(f"{pending['input_name']}_{candidate.candidate_id}.zip"),
            })
            st.success("已加入工作區")

        st.divider()
        st.header("Stage 2｜建立吸附初始結構")
        st.caption(
            "位點是未放鬆初始幾何；最終位置、解離與吸附能必須由 DFT relaxation 判定。"
        )
        ads_source = st.radio(
            "吸附物來源",
            ["內建分子", "上傳 XYZ／POSCAR／CIF"],
            horizontal=True,
            key=f"ads_source_{selected}",
        )
        adsorbate_spec = None
        try:
            if ads_source == "內建分子":
                ads_name = st.selectbox(
                    "吸附物",
                    list(BUILTIN_ADSORBATES),
                    index=list(BUILTIN_ADSORBATES).index("CO"),
                    key=f"ads_name_{selected}",
                )
                template = make_adsorbate(ads_name)
                atom_labels = [
                    f"{index}: {atom.symbol}"
                    for index, atom in enumerate(template.atoms)
                ]
                anchor_index = st.selectbox(
                    "錨定原子",
                    range(len(atom_labels)),
                    index=template.anchor_index,
                    format_func=lambda index: atom_labels[index],
                    key=f"anchor_{selected}_{ads_name}",
                )
                charge = st.number_input(
                    "形式電荷（僅記錄，不自動產生 NELECT）",
                    value=0,
                    step=1,
                    key=f"charge_{selected}_{ads_name}",
                )
                multiplicity = st.number_input(
                    "自旋多重度（僅記錄）",
                    min_value=1,
                    value=int(template.spin_multiplicity or 1),
                    step=1,
                    key=f"spin_{selected}_{ads_name}",
                )
                adsorbate_spec = make_adsorbate(
                    ads_name,
                    anchor_index=anchor_index,
                    charge=charge,
                    spin_multiplicity=multiplicity,
                )
            else:
                ads_upload = st.file_uploader(
                    "上傳吸附物",
                    type=None,
                    key=f"ads_upload_{selected}",
                )
                if ads_upload is not None:
                    uploaded_atoms = read_adsorbate_upload(ads_upload)
                    atom_labels = [
                        f"{index}: {atom.symbol}"
                        for index, atom in enumerate(uploaded_atoms)
                    ]
                    anchor_index = st.selectbox(
                        "錨定原子",
                        range(len(atom_labels)),
                        format_func=lambda index: atom_labels[index],
                        key=f"uploaded_anchor_{selected}",
                    )
                    charge = st.number_input(
                        "形式電荷（僅記錄）",
                        value=0,
                        step=1,
                        key=f"uploaded_charge_{selected}",
                    )
                    multiplicity = st.number_input(
                        "自旋多重度（0 表示未知）",
                        min_value=0,
                        value=0,
                        step=1,
                        key=f"uploaded_spin_{selected}",
                    )
                    adsorbate_spec = make_uploaded_adsorbate(
                        uploaded_atoms,
                        name=ads_upload.name,
                        anchor_index=anchor_index,
                        charge=charge,
                        spin_multiplicity=(multiplicity or None),
                    )

            if adsorbate_spec is not None:
                adsorbate_report = validate_adsorbate(adsorbate_spec)
                st.subheader("Stage 2.1｜吸附物驗證")
                show_mapping_table({
                    "狀態": adsorbate_report.status,
                    "化學式": adsorbate_report.metrics.get("formula"),
                    "原子數": adsorbate_report.metrics.get("atom_count"),
                    "錨定原子": adsorbate_report.metrics.get("anchor_symbol"),
                    "鍵數": adsorbate_report.metrics.get("bond_count"),
                    "電荷／自旋多重度": (
                        f"{adsorbate_report.metrics.get('charge')} / "
                        f"{adsorbate_report.metrics.get('spin_multiplicity')}"
                    ),
                })
                for message in adsorbate_report.errors:
                    st.error(message)
                for message in adsorbate_report.warnings:
                    st.warning(message)

                if adsorbate_report.passed:
                    sites = enumerate_adsorption_sites(
                        candidate.atoms, candidate.metadata, unique=True
                    )
                    st.subheader("Stage 2.2｜週期性吸附位點")
                    st.dataframe([
                        {
                            "選項": site.site_id,
                            "類型": site.site_type,
                            "配位": site.coordination,
                            "週期等價數": site.multiplicity,
                            "表面元素": ", ".join(
                                site.metadata.get("coordinating_symbols", [])
                            ),
                            "fractional xy": (
                                f"{site.fractional_xy[0]:.6f}, "
                                f"{site.fractional_xy[1]:.6f}"
                            ),
                        }
                        for site in sites
                    ], width="stretch", hide_index=True)
                    site_ids = [site.site_id for site in sites]
                    selected_sites = st.multiselect(
                        "要產生的位點",
                        site_ids,
                        default=site_ids,
                        format_func=lambda site_id: next(
                            f"{site.site_id} — {site.site_type}"
                            for site in sites if site.site_id == site_id
                        ),
                        key=f"selected_sites_{selected}",
                    )
                    parameter_columns = st.columns(3)
                    height_text = parameter_columns[0].text_input(
                        "高度 Å（逗號分隔）", "2.0", key=f"ads_heights_{selected}"
                    )
                    tilt_text = parameter_columns[1].text_input(
                        "傾斜角 °", "0", key=f"ads_tilts_{selected}"
                    )
                    azimuth_text = parameter_columns[2].text_input(
                        "方位角 °", "0", key=f"ads_azimuths_{selected}"
                    )

                    if st.button(
                        "產生並驗證吸附候選",
                        type="primary",
                        key=f"generate_ads_{selected}",
                    ):
                        heights = parse_number_list(height_text, "吸附高度", minimum=0.01)
                        tilts = parse_number_list(tilt_text, "傾斜角", minimum=0, maximum=180)
                        azimuths = parse_number_list(azimuth_text, "方位角")
                        ads_report, found_sites, adsorption_results = (
                            generate_adsorption_candidates(
                                slab_candidate=candidate,
                                adsorbate_spec=adsorbate_spec,
                                selected_site_ids=selected_sites,
                                heights_A=heights,
                                tilts_deg=tilts,
                                azimuths_deg=azimuths,
                                vacuum_total_A=vacuum_total,
                            )
                        )
                        st.session_state.adsorption_pending = {
                            "base_candidate_id": candidate.candidate_id,
                            "adsorbate_spec": adsorbate_spec,
                            "results": adsorption_results,
                        }
                        st.success(f"已產生 {len(adsorption_results)} 個吸附候選")
        except Exception as exc:
            st.session_state.adsorption_pending = None
            st.error(f"Stage 2 產生失敗：{exc}")

        adsorption_pending = st.session_state.adsorption_pending
        if (
            adsorption_pending
            and adsorption_pending.get("base_candidate_id") == candidate.candidate_id
        ):
            st.subheader("Stage 2.3｜吸附候選驗證")
            adsorption_labels = [
                f"{ads_candidate.candidate_id}｜{ads_report.status}"
                for ads_candidate, ads_report in adsorption_pending["results"]
            ]
            adsorption_selected = st.selectbox(
                "選擇吸附候選",
                range(len(adsorption_labels)),
                format_func=lambda index: adsorption_labels[index],
                key=f"ads_candidate_select_{selected}",
            )
            ads_candidate, ads_report = adsorption_pending["results"][adsorption_selected]
            render_atoms(
                ads_candidate.atoms,
                f"ads_{selected}_{adsorption_selected}",
                wrap_positions=False,
            )
            adsorption_report_table(ads_report)
            if ads_report.passed:
                ads_package, ads_poscar = make_adsorption_package(
                    candidate,
                    adsorption_pending["adsorbate_spec"],
                    ads_candidate,
                    ads_report,
                )
                st.download_button(
                    "下載吸附候選結構包",
                    data=ads_package,
                    file_name=safe_name(ads_candidate.candidate_id + ".zip"),
                    mime="application/zip",
                    key=f"ads_download_{selected}_{adsorption_selected}",
                )
            else:
                st.error("此吸附候選未通過強制驗證，因此禁止下載。")
    else:
        st.error("此候選未通過強制驗證，因此不提供 POSCAR／ZIP 下載。")

if st.session_state.workspaces:
    st.header("已驗證工作區")
    for index, workspace in enumerate(st.session_state.workspaces):
        with st.expander(workspace["name"], expanded=False):
            st.json(workspace["report"])
            st.download_button(
                "下載",
                data=workspace["package"],
                file_name=workspace["filename"],
                mime="application/zip",
                key=f"workspace_download_{index}",
            )
    if st.button("清空工作區"):
        st.session_state.workspaces = []
        st.session_state.pending = None
        st.rerun()
