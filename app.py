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
from ase.build import surface
from ase.neighborlist import neighbor_list, natural_cutoffs

# 初始化：存放多個工作區
if 'workspaces' not in st.session_state:
    st.session_state.workspaces = []

# --- 🛠️ 晶體結構自動識別邏輯 (極致優化：Dynamic Cutoff + Bond Angle CNA) ---
def crystal_structure_classifier(atoms, tolerance_pct=10):
    try:
        n_atoms = len(atoms)
        if n_atoms == 0:
            return "空結構"
            
        # 1. 導入 alpha 修正係數與容忍度
        alpha = 1.15
        tol_factor = 1.0 + (tolerance_pct / 100.0)
        
        # 2. 使用 ASE 官方推薦的 natural_cutoffs，給予每個原子專屬的搜尋半徑
        cutoffs = natural_cutoffs(atoms, mult=alpha * tol_factor)
        
        # 3. 搜尋鄰居，大幅提升搜尋效率
        first_indices, second_indices, distances = neighbor_list('ijd', atoms, cutoff=cutoffs)
        
        # 4. 利用 np.bincount 計算配位數，避免雙重計算與 i->j 偏差
        cn_counts = np.bincount(first_indices, minlength=n_atoms)
        avg_cn = np.mean(cn_counts) if len(cn_counts) > 0 else 0
        
        # 5. 根據配位數與局部幾何 (Local Geometry) 進行拓撲辨識
        if 11.0 <= avg_cn <= 13.0:
            # 尋找一個配位數 >= 11 的代表性原子來分析第一配位殼層鍵角
            target_i = next((i for i, cn in enumerate(cn_counts) if cn >= 11), -1)
            
            if target_i != -1:
                # 取得目標原子的所有鄰居
                neighbors_of_i = second_indices[first_indices == target_i]
                neighbors_of_i = [j for j in neighbors_of_i if j != target_i]
                
                has_109 = False
                # 取得中心原子到所有鄰居的向量 (考慮週期性邊界 mic=True)
                _, v_ij = atoms.get_distances(target_i, neighbors_of_i, mic=True, vector=True)
                
                # 計算任意兩個相鄰向量之間的夾角
                for x in range(len(v_ij)):
                    for y in range(x+1, len(v_ij)):
                        v1, v2 = v_ij[x], v_ij[y]
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 0 and n2 > 0:
                            cos_theta = np.dot(v1, v2) / (n1 * n2)
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            angle = np.degrees(np.arccos(cos_theta))
                            
                            # 109.5度為 HCP 的特徵鍵角 (FCC不會出現)
                            if 105 < angle < 115:
                                has_109 = True
                                break
                    if has_109:
                        break
                
                if has_109:
                    return f"HCP (六方最密堆積, 平均 CN: {round(avg_cn, 2)})"
                else:
                    return f"FCC (面心立方結構, 平均 CN: {round(avg_cn, 2)})"
            else:
                return f"密集堆積 (平均 CN: {round(avg_cn, 2)})"
        elif 7.0 <= avg_cn <= 9.0:
            return f"BCC (體心立方結構, 平均 CN: {round(avg_cn, 2)})"
        elif 5.0 <= avg_cn <= 6.5:
            return f"SC (簡單立方結構, 平均 CN: {round(avg_cn, 2)})"
        else:
            return f"低對稱或弛豫畸變結構 (平均 CN: {round(avg_cn, 2)})"
            
    except Exception as e:
        traceback.print_exc() # 將錯誤印在後台以便 Debug
        return f"識別演算法異常 ({str(e)})"

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

# --- 側邊欄：參數設定 ---
st.sidebar.header("⚙️ 參數設定")
st.sidebar.markdown("**1. 晶面 (h, k, l)**")
col1, col2, col3 = st.sidebar.columns(3)
h = col1.number_input("h", value=1, step=1)
k = col2.number_input("k", value=1, step=1)
l = col3.number_input("l", value=1, step=1)

layers = st.sidebar.number_input("Slab 層數 (Layers)", value=10, min_value=1)
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
st.sidebar.markdown("**3. 🚀 表面計算優化設定**")
fix_ratio = st.sidebar.slider("固定底部原子比例 (%)", min_value=0, max_value=100, value=40, step=5)
st.sidebar.caption("💡 提示：表面計算通常固定底部 40% ~ 50% 的原子以模擬內部 bulk 環境。")

st.sidebar.markdown("---")
st.sidebar.markdown("**4. 🔬 晶體識別幾何優化**")
tolerance_pct = st.sidebar.slider("晶格變形容忍度 (%)", min_value=5, max_value=20, value=10, step=1)
st.sidebar.caption("💡 說明：配合動態矩陣 α=1.15，預設 10% 已能精準捕捉第一殼層，過高易誤納第二殼層。")

st.sidebar.markdown("---")

# --- 主程式：上傳區 ---
uploaded_files = st.file_uploader("上傳金屬 POSCAR (支援多檔案同時拖曳)", key="uploader", accept_multiple_files=True)

if uploaded_files:
    if st.button("➕ 批次新增到工作區", type="primary"):
        for uploaded_file in uploaded_files:
            try:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                bulk = read_vasp(stringio)
                
                # 呼叫升級後的動態容忍度分類器
                detected_lattice = crystal_structure_classifier(bulk, tolerance_pct=tolerance_pct)
                
                if (h, k, l) == (0, 0, 0):
                    raise ValueError("Miller index 不可為 (0, 0, 0)")

                # vacuum_total 是週期影像之間的總真空；ASE 的 vacuum 參數是單側厚度。
                # 下載用結構不因 Viewer 需求而 wrap、平移或複製。
                slab = surface(
                    bulk,
                    (h, k, l),
                    layers=layers,
                    vacuum=vacuum_total / 2.0,
                )
                if super_x > 1 or super_y > 1:
                    slab = slab * (super_x, super_y, 1)
                
                # --- 結構體檢優化：距離 < 0.6*(Ri+Rj) 視為重疊 ---
                overlap_cutoffs = natural_cutoffs(slab, mult=0.6)
                i_over, j_over, d_over = neighbor_list('ijd', slab, cutoff=overlap_cutoffs)
                overlaps = [dist for idx_i, idx_j, dist in zip(i_over, j_over, d_over) if idx_i != idx_j]
                health_status = "✅ 正常" if len(overlaps) == 0 else "⚠️ 異常(原子重疊)"
                
                # Selective Dynamics 邏輯
                z_positions = slab.positions[:, 2]
                z_min, z_max = np.min(z_positions), np.max(z_positions)
                z_range = z_max - z_min if (z_max - z_min) > 0 else 1.0
                
                flags = []
                for pos in slab.positions:
                    relative_height = (pos[2] - z_min) / z_range * 100
                    if relative_height <= fix_ratio:
                        flags.append([False, False, False]) # Fixed
                    else:
                        flags.append([True, True, True])   # Relaxed
                
                from ase.constraints import FixAtoms
                fixed_indices = [idx for idx, flag in enumerate(flags) if not flag[0]]
                slab.set_constraint(FixAtoms(indices=fixed_indices))
                
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
                
                # 1. 嚴格依照 PDF 第 8 頁格式
                info_content = f"""Input file: {uploaded_file.name}
Miller index: ({h}, {k}, {l})
Layers: {layers}
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
   - 晶體結構體檢: {detected_lattice}
2. 切割晶面: 沿著 ({h}, {k}, {l}) 方向，厚度為 {layers} 層。
3. 增加真空層: 週期影像間總真空約 {vacuum_total} Å（上下各約 {vacuum_total / 2:.2f} Å）。
4. 擴張表面積: 建立 {super_x}x{super_y}x1 Supercell。
5. Selective Dynamics (底層固定):
   - 固定底部 {fix_ratio}% 的原子以模擬 bulk 內部。
6. 結構異常檢查: {health_status}。
--------------------------------------------------
"""
                
                # --- 幾何物理計算優化 ---
                # 表面積改為向量叉積 (支援非正交晶胞)
                cross_prod = np.cross(slab.cell[0], slab.cell[1])
                true_area = np.linalg.norm(cross_prod)
                
                # 真空比例改為真實物理真空空間
                # 晶胞沿表面法向的高度：體積 / 表面積，也適用於傾斜晶胞。
                total_height = abs(slab.get_volume()) / true_area if true_area > 0 else 0
                slab_thickness = z_max - z_min
                actual_vacuum = total_height - slab_thickness
                vacuum_ratio = (actual_vacuum / total_height) * 100 if total_height > 0 else 0
                
                num_fixed = len(fixed_indices)
                num_relaxed = len(slab) - num_fixed
                
                summary = {
                    "原始晶體結構": detected_lattice,
                    "晶面 (hkl)": f"({h}, {k}, {l})",
                    "層數": layers,
                    "Supercell": f"{super_x}x{super_y}x1",
                    "原子總數": f"{len(slab)} (固定: {num_fixed} / 放鬆: {num_relaxed})",
                    "表面積": f"{round(true_area, 2)} Å²",
                    "真空層佔比": f"{round(vacuum_ratio, 1)}% ({round(actual_vacuum, 2)} Å)"
                }
                
                new_ws = {
                    "name": uploaded_file.name.split('.')[0],
                    "filename": f"POSCAR_{uploaded_file.name.split('.')[0]}_{h}{k}{l}",
                    "poscar": poscar_data,
                    "viewer_xyz": viewer_xyz,
                    "viewer_cell": np.asarray(slab.cell).tolist(),
                    "viewer_shift": viewer_shift.tolist(),
                    "info_txt": info_content,  
                    "log_txt": log_content,    
                    "summary": summary,
                    "health": health_status,
                    "hkl": f"{h}{k}{l}"
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
            
            c1, c2 = st.columns(2)
            show_bonds = c1.checkbox("🔗 實體金屬鍵 (Ball-and-stick)", value=True, key=f"b_{i}")
            show_box = c2.checkbox("📦 銳利晶界框線 (Unit Cell)", value=True, key=f"box_{i}")
            
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
            
            # 🚀 唯一被允許存在的主畫面繪圖指令，杜絕多圖框幽靈
            showmol(view, height=450, width=700)
            
            st.subheader("📊 結構資訊摘要")
            st.table(ws["summary"])
            
            col_act1, col_act2 = st.columns([0.5, 0.5])
            with col_act1:
                single_zip = io.BytesIO()
                with zipfile.ZipFile(single_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    folder_name = f"{ws['name']}_{ws['hkl']}_output/"
                    
                    zf.writestr(f"{folder_name}POSCAR", ws["poscar"])
                    zf.writestr(f"{folder_name}slab_info.txt", ws["info_txt"])
                    zf.writestr(f"{folder_name}README_or_log.txt", ws["log_txt"])
                    
                html_link = create_download_link(single_zip.getvalue(), f"{ws['filename']}_output.zip", "📥 下載合規結構包 (ZIP)", is_zip=True)
                st.markdown(html_link, unsafe_allow_html=True)
            with col_act2:
                st.info("💡 提示：若要關閉此結構分頁，請點擊左側側邊欄「已載入結構清單」中該結構旁邊的❌。")
                
else:
    st.info("請在上方上傳一或多個金屬 POSCAR 檔案，並點擊「批次新增到工作區」。")
