import streamlit as st
import py3Dmol
import io
import zipfile
import base64
import numpy as np
import traceback
from stmol import showmol
from ase.io import write
from ase.io.vasp import read_vasp
from slab_core import (
    apply_bottom_plane_constraint,
    build_from_bulk,
    describe_input_cell,
    periodic_gap_analysis,
    prepare_existing_slab,
    validate_slab_geometry,
)

APP_VERSION = "2026-07-16-stage1-v1"

# 初始化：存放多個工作區
if 'workspaces' not in st.session_state:
    st.session_state.workspaces = []

# --- 🛠️ HTML 下載按鈕 ---
def create_download_link(data, filename, button_text, is_zip=False):
    if is_zip:
        b64 = base64.b64encode(data).decode()
        mime = "application/zip"
    else:
        b64 = base64.b64encode(data.encode('utf-8')).decode()
        mime = "text/plain"
    
    css = """
    <style>
    .custom-dl-btn {
        background-color: #FF4B4B; color: white; padding: 0.5rem 1rem;
        border-radius: 0.5rem; text-decoration: none; display: inline-block;
        font-weight: 500; margin-top: 10px; text-align: center; width: 90%;
    }
    .custom-dl-btn:hover { background-color: #FF6666; color: white; }
    </style>
    """
    return f'{css}<a href="data:{mime};base64,{b64}" download="{filename}" class="custom-dl-btn">{button_text}</a>'

# --- 🛠️ 注入 CSS 讓分頁中的「X」按鈕變好看 ---
st.markdown(
    """
    <style>
        div.stTabs [data-testid="stMarkdownContainer"] button {
            background-color: transparent !important;
            border: none !important;
            color: #AAAAAA !important;
            cursor: pointer;
            font-size: 14px !important;
            font-weight: normal !important;
            margin-left: 8px !important;
            padding: 2px 5px !important;
            vertical-align: middle;
        }
        div.stTabs [data-testid="stMarkdownContainer"] button:hover {
            background-color: #FFDDDD !important;
            border-radius: 4px;
            color: #FF4B4B !important;
        }
        div.stTabs [aria-selected="true"] button {
            color: #555555 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Slab Builder 研究室版 (VESTA 批次管理版) 🔬")
st.caption(f"版本：{APP_VERSION}")

# --- 側邊欄：參數設定 ---
st.sidebar.header("⚙️ 參數設定")
input_mode = st.sidebar.radio(
    "輸入處理模式",
    ["自動判斷", "由 3D Bulk 建立新 Slab", "處理既有 Slab"],
    help="已有真空的 slab 不可再次執行 surface()。自動判斷遇到疑似 slab 時會停止並要求確認。",
)

h = k = l = None
layers = None
allow_slab_as_bulk = False
if input_mode != "處理既有 Slab":
    st.sidebar.markdown("**1. 晶面 (h, k, l)**")
    col1, col2, col3 = st.sidebar.columns(3)
    h = col1.number_input("h", value=1, step=1)
    k = col2.number_input("k", value=1, step=1)
    l = col3.number_input("l", value=1, step=1)
    layers = st.sidebar.number_input(
        "ASE 結構重複層數",
        value=10,
        min_value=1,
        help="此數值不一定等於實際原子平面數；產生後會另外計算並顯示。",
    )
    if input_mode == "由 3D Bulk 建立新 Slab":
        allow_slab_as_bulk = st.sidebar.checkbox(
            "進階：仍將疑似 Slab 當作 Bulk",
            value=False,
            help="只有確認偵測為誤判時才啟用；一般情況不建議。",
        )
else:
    st.sidebar.info("既有 Slab 模式不會重新切晶面，只調整真空、表面超晶胞與固定層。")
vacuum_total = st.sidebar.number_input(
    "週期影像間的總真空距離 (Å)",
    value=15.00,
    min_value=0.1,
    help="ASE 會把此數值的一半放在 slab 上方、另一半放在下方。",
)
st.sidebar.caption(
    f"目前設定：上下各約 {vacuum_total / 2:.2f} Å，總真空約 {vacuum_total:.2f} Å。"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**2. 擴張表面積 (Supercell)**")
col_s1, col_s2 = st.sidebar.columns(2)
super_x = col_s1.number_input("X 倍數", value=3, min_value=1)
super_y = col_s2.number_input("Y 倍數", value=3, min_value=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**3. 🚀 Selective Dynamics 設定**")
fixed_bottom_planes = st.sidebar.number_input(
    "固定底部原子平面數",
    value=4,
    min_value=0,
    step=1,
    help="程式會辨識原子平面並整層固定，不會把同一層切成固定與放鬆兩部分。",
)

st.sidebar.markdown("---")

# --- 主程式：上傳區 ---
uploaded_files = st.file_uploader("上傳金屬 POSCAR (支援多檔案同時拖曳)", key="uploader", accept_multiple_files=True)

if uploaded_files:
    input_checks = []
    for uploaded_file in uploaded_files:
        try:
            preview_atoms = read_vasp(
                io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            )
            preview_gap = periodic_gap_analysis(preview_atoms)
            input_checks.append({
                "檔案": uploaded_file.name,
                "輸入晶胞": preview_atoms.get_chemical_formula(),
                "最大週期空白": f"{preview_gap['largest_gap']:.2f} Å",
                "判定": (
                    "疑似既有 Slab"
                    if preview_gap["likely_slab"]
                    else "可作為 3D Bulk 候選"
                ),
            })
        except Exception as exc:
            input_checks.append({
                "檔案": uploaded_file.name,
                "輸入晶胞": "讀取失敗",
                "最大週期空白": "—",
                "判定": str(exc),
            })
    st.subheader("輸入結構預檢")
    st.dataframe(input_checks, use_container_width=True, hide_index=True)

    if st.button("➕ 批次新增到工作區", type="primary"):
        for uploaded_file in uploaded_files:
            try:
                input_text = uploaded_file.getvalue().decode("utf-8")
                source_atoms = read_vasp(io.StringIO(input_text))
                input_description = describe_input_cell(source_atoms)
                input_gap = periodic_gap_analysis(source_atoms)

                if input_mode == "自動判斷":
                    if input_gap["likely_slab"]:
                        raise ValueError(
                            f"偵測到約 {input_gap['largest_gap']:.2f} Å 的週期空白，"
                            "輸入可能已是 slab。請改選『處理既有 Slab』；"
                            "若要建立新晶面，請改上傳 3D bulk conventional cell。"
                        )
                    effective_mode = "bulk"
                elif input_mode == "由 3D Bulk 建立新 Slab":
                    if input_gap["likely_slab"] and not allow_slab_as_bulk:
                        raise ValueError(
                            f"輸入含約 {input_gap['largest_gap']:.2f} Å 大空白，"
                            "已阻止 slab 再次切割。只有確認偵測誤判時才啟用進階覆寫。"
                        )
                    effective_mode = "bulk"
                else:
                    effective_mode = "existing_slab"

                if effective_mode == "bulk":
                    slab, reduced_hkl = build_from_bulk(
                        source_atoms,
                        (h, k, l),
                        layers,
                        vacuum_total,
                        (super_x, super_y),
                    )
                    hkl_display = f"({reduced_hkl[0]}, {reduced_hkl[1]}, {reduced_hkl[2]})"
                    hkl_key = "".join(str(v) for v in reduced_hkl)
                    workflow_description = (
                        f"由 3D bulk 沿 {hkl_display} 切割；"
                        f"ASE 結構重複層數 {layers}"
                    )
                else:
                    slab = prepare_existing_slab(
                        source_atoms,
                        vacuum_total,
                        (super_x, super_y),
                    )
                    hkl_display = "沿用既有 Slab"
                    hkl_key = "existing"
                    workflow_description = "沿用既有 slab；未重新執行 surface()"

                atomic_planes, applied_fixed_planes, fixed_indices = (
                    apply_bottom_plane_constraint(slab, fixed_bottom_planes)
                )
                geometry = validate_slab_geometry(slab, planes=atomic_planes)
                overlap = geometry["overlap"]
                health_status = (
                    "✅ 未發現嚴重原子重疊"
                    if not overlap["has_overlap"]
                    else "⚠️ 發現疑似嚴重原子重疊"
                )
                
                out_buffer = io.StringIO()
                write(out_buffer, slab, format="vasp", vasp5=True)
                poscar_data = out_buffer.getvalue()

                # Viewer 使用獨立、無週期的 XYZ 副本，避免 py3Dmol 依 VASP
                # 晶胞推斷跨週期鍵結或顯示重複影像。只改畫面，不改 POSCAR。
                viewer_slab = slab.copy()
                viewer_shift = -0.5 * np.sum(np.asarray(slab.cell), axis=0)
                viewer_slab.translate(viewer_shift)
                viewer_slab.set_pbc(False)
                viewer_buffer = io.StringIO()
                write(viewer_buffer, viewer_slab, format="xyz")
                viewer_xyz = viewer_buffer.getvalue()
                
                import datetime
                current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                info_content = f"""Input file: {uploaded_file.name}
Input mode: {effective_mode}
Surface: {hkl_display}
ASE structure layers: {layers if layers is not None else 'not applied'}
Actual atomic planes: {geometry['plane_count']}
Total vacuum gap: {vacuum_total} Å (approximately {vacuum_total / 2:.2f} Å per side)
Supercell: {super_x} x {super_y} x 1
Output file: POSCAR
Generated by: Streamlit Slab Builder
Package: ASE
Date: {current_date}
"""

                # 2. 說明生成結果與體檢報告
                log_content = f"""[LOG] Slab Generation Workflow
--------------------------------------------------
本結果由網頁端工具自動化流水線產生。
1. 讀取原始結構: {uploaded_file.name}
   - 輸入晶胞: {input_description}
   - 輸入最大週期空白: {input_gap['largest_gap']:.4f} Å
2. 結構處理: {workflow_description}
   - 實際辨識原子平面數: {geometry['plane_count']}
3. 設定真空層: 輸出週期影像間實測空白 {geometry['vacuum_gap']:.4f} Å。
4. 擴張表面積: 建立 {super_x}x{super_y}x1 Supercell。
5. Selective Dynamics (底層固定):
   - 固定底部 {applied_fixed_planes} 個完整原子平面，共 {len(fixed_indices)} 個原子。
6. 幾何檢查:
   - {health_status}
   - 最短週期原子距離: {geometry['minimum_distance']:.4f} Å
   - Slab 厚度: {geometry['slab_thickness']:.4f} Å
--------------------------------------------------
"""

                num_fixed = len(fixed_indices)
                num_relaxed = len(slab) - num_fixed
                
                summary = {
                    "輸入晶胞": input_description,
                    "輸入判定": "疑似既有 Slab" if input_gap["likely_slab"] else "3D Bulk 候選",
                    "處理方式": workflow_description,
                    "晶面 (hkl)": hkl_display,
                    "ASE 結構重複層數": layers if layers is not None else "未套用",
                    "實際原子平面數": geometry["plane_count"],
                    "每個平面原子數": ", ".join(str(len(p)) for p in atomic_planes),
                    "Supercell": f"{super_x}x{super_y}x1",
                    "原子總數": f"{len(slab)} (固定: {num_fixed} / 放鬆: {num_relaxed})",
                    "固定完整平面數": applied_fixed_planes,
                    "最短週期原子距離": f"{geometry['minimum_distance']:.4f} Å",
                    "Slab 厚度": f"{geometry['slab_thickness']:.4f} Å",
                    "表面積": f"{geometry['surface_area']:.2f} Å²",
                    "實測總真空": f"{geometry['vacuum_gap']:.4f} Å",
                }
                
                new_ws = {
                    "name": uploaded_file.name.split('.')[0],
                    "filename": f"POSCAR_{uploaded_file.name.split('.')[0]}_{hkl_key}",
                    "input_poscar": input_text,
                    "poscar": poscar_data,
                    "viewer_xyz": viewer_xyz,
                    "viewer_cell": np.asarray(slab.cell).tolist(),
                    "viewer_shift": viewer_shift.tolist(),
                    "info_txt": info_content,  
                    "log_txt": log_content,    
                    "summary": summary,
                    "health": health_status,
                    "hkl": hkl_key,
                }
                st.session_state.workspaces.append(new_ws)
            except Exception as e:
                traceback.print_exc()
                st.sidebar.error(f"檔案 {uploaded_file.name} 處理失敗: {e}")
        st.rerun()

# --- 側邊欄：🗂️ 已載入結構清單 ---
selected_indices = []
if st.session_state.workspaces:
    st.sidebar.markdown("### 🗂️ 已載入結構清單")
    for idx, ws in enumerate(st.session_state.workspaces):
        col_side1, col_side2 = st.sidebar.columns([0.85, 0.15])
        label = f"[{ws['health']}] {ws['name']} - ({ws['hkl']})"
        is_selected = col_side1.checkbox(label, value=True, key=f"select_side_{idx}")
        if is_selected:
            selected_indices.append(idx)
        
        if col_side2.button("❌", key=f"del_side_{idx}"):
            st.session_state.workspaces.pop(idx)
            st.rerun()
            
    st.sidebar.markdown(" ")
    if selected_indices:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx in selected_indices:
                ws = st.session_state.workspaces[idx]
                folder_name = f"{ws['name']}_{ws['hkl']}_output/"
                
                zip_file.writestr(f"{folder_name}input_POSCAR", ws.get("input_poscar", ""))
                zip_file.writestr(f"{folder_name}POSCAR", ws["poscar"])
                zip_file.writestr(f"{folder_name}slab_info.txt", ws["info_txt"])
                zip_file.writestr(f"{folder_name}README_or_log.txt", ws["log_txt"])
        
        html_link = create_download_link(zip_buffer.getvalue(), "batch_slab_outputs.zip", "📦 批次下載合規資料夾 (ZIP)", is_zip=True)
        st.sidebar.markdown(html_link, unsafe_allow_html=True)
        st.sidebar.markdown(" ")
    
    if st.sidebar.button("🗑️ 清空所有工作區", type="secondary", use_container_width=True):
        st.session_state.workspaces = []
        st.rerun()

# --- 主畫面：分頁管理 ---
if st.session_state.workspaces:
    clean_titles = [f"{ws['name']} - ({ws['hkl']})" for ws in st.session_state.workspaces]
    tabs = st.tabs(clean_titles)

    for i, tab in enumerate(tabs):
        with tab:
            ws = st.session_state.workspaces[i]
            
            c1, c2, c3 = st.columns(3)
            show_bonds = c1.checkbox("🔗 實體金屬鍵 (Ball-and-stick)", value=True, key=f"b_{i}")
            show_box = c2.checkbox("📦 銳利晶界框線 (Unit Cell)", value=True, key=f"box_{i}")
            view_orientation = c3.selectbox(
                "觀看方向",
                ["立體", "俯視", "側視"],
                key=f"orientation_{i}",
            )
            
            # --- 💎 3D 視覺化 (修復重複出框與原子隱形 Bug 版) ---
            view = py3Dmol.view(width=700, height=450)

            # 舊 Session 內的工作區可能尚未含 viewer_xyz；即時轉換即可，
            # 不需要修改真正下載的 POSCAR。
            if "viewer_xyz" not in ws:
                old_slab = read_vasp(io.StringIO(ws["poscar"]))
                old_shift = -0.5 * np.sum(np.asarray(old_slab.cell), axis=0)
                old_slab.translate(old_shift)
                old_slab.set_pbc(False)
                old_viewer_buffer = io.StringIO()
                write(old_viewer_buffer, old_slab, format="xyz")
                ws["viewer_xyz"] = old_viewer_buffer.getvalue()
                ws["viewer_cell"] = np.asarray(old_slab.cell).tolist()
                ws["viewer_shift"] = old_shift.tolist()

            view.addModel(ws["viewer_xyz"], "xyz")
            
            style = {
                'sphere': {
                    'colorscheme': 'Jmol', 
                    'scale': 0.30, 
                    'outline': {'color': '#333333', 'width': 0.04}
                }
            }
            if show_bonds: 
                style['stick'] = {
                    'colorscheme': 'Jmol', 
                    'radius': 0.10
                }
            view.setStyle(style)
            
            if show_box:
                # XYZ 模型沒有週期性；以 12 條線獨立畫出同一個晶胞。
                # 如此可顯示真空高度，又不會觸發週期影像或跨邊界鍵結。
                cell = np.asarray(ws["viewer_cell"], dtype=float)
                shift = np.asarray(ws["viewer_shift"], dtype=float)
                origin = shift
                a, b, c = cell
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
                    (0, 1), (0, 2), (0, 3),
                    (1, 4), (1, 5),
                    (2, 4), (2, 6),
                    (3, 5), (3, 6),
                    (4, 7), (5, 7), (6, 7),
                ]
                for start_idx, end_idx in edges:
                    start = corners[start_idx]
                    end = corners[end_idx]
                    view.addLine({
                        "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
                        "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
                        "color": "#444444",
                        "linewidth": 1.5,
                    })
                
            view.setBackgroundColor('#F8F9FA') 
            view.zoomTo()
            if view_orientation == "俯視":
                # surface() 的表面法向為 z；identity quaternion 沿 z 俯視。
                view.setView([0, 0, 0, 0, 0, 0, 0, 1])
                view.zoomTo()
            elif view_orientation == "側視":
                view.setView([0, 0, 0, 0, 0, 0, 0, 1])
                view.rotate(90, "x")
                view.zoomTo()
            else:
                view.rotate(-55, "x")
                view.rotate(30, "z")
            
            # 🚀 唯一被允許存在的主畫面繪圖指令，杜絕多圖框幽靈
            showmol(view, height=450, width=700)
            
            st.subheader("📊 結構資訊摘要")
            st.table(ws["summary"])
            
            col_act1, col_act2 = st.columns([0.5, 0.5])
            with col_act1:
                single_zip = io.BytesIO()
                with zipfile.ZipFile(single_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    folder_name = f"{ws['name']}_{ws['hkl']}_output/"
                    
                    zf.writestr(f"{folder_name}input_POSCAR", ws.get("input_poscar", ""))
                    zf.writestr(f"{folder_name}POSCAR", ws["poscar"])
                    zf.writestr(f"{folder_name}slab_info.txt", ws["info_txt"])
                    zf.writestr(f"{folder_name}README_or_log.txt", ws["log_txt"])
                    
                html_link = create_download_link(single_zip.getvalue(), f"{ws['filename']}_output.zip", "📥 下載合規結構包 (ZIP)", is_zip=True)
                st.markdown(html_link, unsafe_allow_html=True)
            with col_act2:
                st.info("💡 提示：若要關閉此結構分頁，請點擊左側側邊欄「已載入結構清單」中該結構旁邊的❌。")
                
else:
    st.info("請在上方上傳一或多個金屬 POSCAR 檔案，並點擊「批次新增到工作區」。")
