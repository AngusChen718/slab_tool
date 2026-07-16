import datetime
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from ase.io import read, write

from slab_core import assess_input, classify_bulk, generate_candidates


APP_VERSION = "2026-07-17-stage1.3-v1"


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
        "Slab Builder Stage 1.3\n"
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


def render_atoms(atoms, widget_key, height=470):
    controls = st.columns(3)
    show_bonds = controls[0].checkbox("顯示鍵結", True, key=f"bonds_{widget_key}")
    show_cell = controls[1].checkbox("顯示晶胞", True, key=f"cell_{widget_key}")
    orientation = controls[2].selectbox(
        "觀看方向", ["側視", "俯視", "立體"], key=f"view_{widget_key}"
    )

    display_atoms = atoms.copy()
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
    components.html(view._make_html(), height=height, width=780, scrolling=False)


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
    st.table(rows)
    for error in report.errors:
        st.error(error)
    for warning in report.warnings:
        st.warning(warning)


st.set_page_config(page_title="Slab Builder Stage 1.3", layout="wide")
st.title("Slab Builder — Physical Validation Edition")
st.caption(f"版本：{APP_VERSION}｜驗證失敗的候選不可下載")

if "workspaces" not in st.session_state:
    st.session_state.workspaces = []
if "pending" not in st.session_state:
    st.session_state.pending = None

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
        st.table(preview)
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
        st.success(f"已產生 {len(results)} 個候選")
    except Exception as exc:
        st.session_state.pending = None
        st.error(str(exc))

if uploaded is None:
    st.session_state.pending = None
pending = st.session_state.pending
if pending:
    st.header("候選結構")
    labels = []
    for candidate, report in pending["results"]:
        top = candidate.metadata.get("top_composition", {})
        bottom = candidate.metadata.get("bottom_composition", {})
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
