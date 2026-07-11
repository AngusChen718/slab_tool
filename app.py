import streamlit as st
import py3Dmol
import io
import zipfile
import base64
import numpy as np
from stmol import showmol
from ase.io import read, write
from ase.io.vasp import read_vasp
from ase.build import surface
from ase.neighborlist import neighbor_list

# 初始化：存放多個工作區
if 'workspaces' not in st.session_state:
    st.session_state.workspaces = []

# --- 🛠️ 晶體結構自動識別邏輯 (極致優化：動態共價半徑矩陣 + 容忍度控制版) ---
def crystal_structure_classifier(atoms, tolerance_pct=20):
    try:
        from ase.data import covalent_radii
        
        n_atoms = len(atoms)
        if n_atoms == 0:
            return "空結構"
            
        # 1. 取得所有原子的共價半徑列表
        atomic_numbers = atoms.get_atomic_numbers()
        radii = [covalent_radii[z] for z in atomic_numbers]
        
        # 2. 建立動態變形容忍度係數
        tolerance_factor = 1.0 + (tolerance_pct / 100.0)
        
        # 3. 利用 ASE neighbor_list 抓出週期性邊界下的所有原子對與距離
        # 尋找可能的最大 Cutoff 作為鄰居搜尋的臨界上限
        max_cutoff = max(radii) * 2.0 * tolerance_factor
        first_indices, second_indices, distances = neighbor_list('ijd', atoms, cutoff=max_cutoff)
        
        # 4. 遍歷鄰居結果，根據兩兩原子的元素半徑動態過濾出化學鍵
        cn_counts = np.zeros(n_atoms, dtype=int)
        for i, j, dist in zip(first_indices, second_indices, distances):
            if i == j and dist < 0.1:  # 排除原子自身
                continue
            # 關鍵優化：為每一對原子對 (i, j) 計算專屬的臨界距離
            dynamic_cutoff = (radii[i] + radii[j]) * tolerance_factor
            if dist <= dynamic_cutoff:
                cn_counts[i] += 1
                
        avg_cn = np.mean(cn_counts) if len(cn_counts) > 0 else 0
        cell = atoms.get_cell()
        
        # 5. 根據升級後的平均配位數進行更準確的幾何分類
        if 11.0 <= avg_cn <= 13.0:
            # 兼容新舊版 ASE：直接從 cell 矩陣計算長度與夾角
            try:
                # 取得三個晶格向量
                a, b, c = cell[0], cell[1], cell[2]
                # 計算向量長度
                la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
                # 計算 alpha, beta, gamma 夾角 (度)
                alpha = np.degrees(np.arccos(np.dot(b, c) / (lb * lc))) if lb*lc > 0 else 90
                beta  = np.degrees(np.arccos(np.dot(a, c) / (la * lc))) if la*lc > 0 else 90
                gamma = np.degrees(np.arccos(np.dot(a, b) / (la * lb))) if la*lb > 0 else 90
                angles = [alpha, beta, gamma]
            except Exception:
                angles = [90, 90, 90] # 降級處理
                
            if any(np.isclose(a, 120, atol=4) or np.isclose(a, 60, atol=4) for a in angles):
                return f"HCP (六方最密堆積, 平均 CN: {round(avg_cn, 2)})"
            return f"FCC (面心立方結構, 平均 CN: {round(avg_cn, 2)})"
        elif 7.0 <= avg_cn <= 9.0:
            return f"BCC (體心立方結構, 平均 CN: {round(avg_cn, 2)})"
        elif 5.0 <= avg_cn <= 6.5:
            return f"SC (簡單立方結構, 平均 CN: {round(avg_cn, 2)})"
        else:
            return f"低對稱或弛豫畸變結構 (平均 CN: {round(avg_cn, 2)})"
    except Exception as e:
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

# --- 🛠️ 注入 CSS 讓分頁中的「X」按鈕變好看 (模仿瀏覽器分頁) ---
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
vacuum = st.sidebar.number_input("真空層厚度 (Vacuum)", value=15.00, min_value=0.0)

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
tolerance_pct = st.sidebar.slider("晶格變形容忍度 (%)", min_value=5, max_value=40, value=20, step=5)
st.sidebar.caption("💡 說明：調高容忍度可避免因表面嚴重弛豫（Relaxation）或晶格形變導致的配位數誤判。")

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
                
                # 切割與放大
                slab = surface(bulk, (h, k, l), layers=layers, vacuum=vacuum)
                if super_x > 1 or super_y > 1: slab = slab * (super_x, super_y, 1)
                slab.wrap()
                
                # 結構體檢
                dist = neighbor_list('d', slab, cutoff=1.2)
                health_status = "✅ 正常" if len(dist) == 0 else "⚠️ 異常(原子重疊)"
                
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
                
                # --- 🛠️ 嚴格對齊 PDF 規範：生成紀錄文字檔 ---
                import datetime
                current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 1. 嚴格依照 PDF 第 8 頁格式的 slab_info.txt
                info_content = f"""Input file: {uploaded_file.name}
Miller index: ({h}, {k}, {l})
Layers: {layers}
Vacuum: {vacuum} Å
Supercell: {super_x} x {super_y} x 1
Output file: POSCAR
Generated by: Streamlit Slab Builder
Package: ASE
Date: {current_date}
"""

                # 2. 說明生成結果與體檢報告的 README_or_log.txt
                log_content = f"""[LOG] Slab Generation Workflow
--------------------------------------------------
本結果由網頁端工具自動化流水線產生。
1. 讀取原始結構: {uploaded_file.name}
   - 晶體結構體檢: {detected_lattice}
2. 切割晶面: 沿著 ({h}, {k}, {l}) 方向，厚度為 {layers} 層。
3. 增加真空層: 於 z 軸方向加入 {vacuum} Å 真空層。
4. 擴張表面積: 建立 {super_x}x{super_y}x1 Supercell。
5. Selective Dynamics (底層固定):
   - 固定底部 {fix_ratio}% 的原子以模擬 bulk 內部。
6. 結構異常檢查: {health_status}。
--------------------------------------------------
"""
                
                total_height = slab.cell[2,2]
                vacuum_ratio = (vacuum / total_height) * 100
                
                num_fixed = len(fixed_indices)
                num_relaxed = len(slab) - num_fixed
                
                summary = {
                    "原始晶體結構": detected_lattice,
                    "晶面 (hkl)": f"({h}, {k}, {l})",
                    "層數": layers,
                    "Supercell": f"{super_x}x{super_y}x1",
                    "原子總數": f"{len(slab)} (固定: {num_fixed} / 放鬆: {num_relaxed})",
                    "表面積": f"{round(slab.cell[0,0] * slab.cell[1,1], 2)} Å²",
                    "真空層佔比": f"{round(vacuum_ratio, 1)}% ({round(total_height, 2)} Å)"
                }
                
                new_ws = {
                    "name": uploaded_file.name.split('.')[0],
                    "filename": f"POSCAR_{uploaded_file.name.split('.')[0]}_{h}{k}{l}",
                    "poscar": poscar_data,
                    "info_txt": info_content,  
                    "log_txt": log_content,    
                    "summary": summary,
                    "health": health_status,
                    "hkl": f"{h}{k}{l}"
                }
                st.session_state.workspaces.append(new_ws)
            except Exception as e:
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
                # 建立 PDF 要求的資料夾結構 (例如: Pt_111_output/)
                folder_name = f"{ws['name']}_{ws['hkl']}_output/"
                
                # 將三個檔案塞入該資料夾中
                zip_file.writestr(f"{folder_name}POSCAR", ws["poscar"])
                zip_file.writestr(f"{folder_name}slab_info.txt", ws["info_txt"])
                zip_file.writestr(f"{folder_name}README_or_log.txt", ws["log_txt"])
        
        html_link = create_download_link(zip_buffer.getvalue(), "batch_slab_outputs.zip", "📦 批次下載合規資料夾 (ZIP)", is_zip=True)
        st.sidebar.markdown(html_link, unsafe_allow_html=True)
        st.sidebar.markdown(" ")
    
    if st.sidebar.button("🗑️ 清空所有工作區", type="secondary", use_container_width=True):
        st.session_state.workspaces = []
        st.rerun()

# --- 主畫面：分頁管理（VESTA 高階渲染 + 整合叉叉關閉功能） ---
if st.session_state.workspaces:
    clean_titles = [f"{ws['name']} - ({ws['hkl']})" for ws in st.session_state.workspaces]
    tabs = st.tabs(clean_titles)

    for i, tab in enumerate(tabs):
        with tab:
            ws = st.session_state.workspaces[i]
            
            # 1. 💎 3D 視覺化
            c1, c2 = st.columns(2)
            show_bonds = c1.checkbox("🔗 實體金屬鍵 (Ball-and-stick)", value=True, key=f"b_{i}")
            show_box = c2.checkbox("📦 銳利晶界框線 (Unit Cell)", value=True, key=f"box_{i}")
            
            view = py3Dmol.view(width=700, height=450)
            view.addModel(ws["poscar"], 'vasp')
            
            style = {'sphere': {'colorscheme': 'Jmol', 'scale': 0.32}}
            if show_bonds: 
                style['stick'] = {'colorscheme': 'Jmol', 'radius': 0.12}
            view.setStyle(style)
            
            if show_box: 
                view.addUnitCell({'color': '#222222', 'linewidth': 2})
                
            view.setBackgroundColor('white') 
            view.zoomTo()
            showmol(view, height=450, width=700)
            
            # 2. 結構資訊摘要表
            st.subheader("📊 結構資訊摘要")
            st.table(ws["summary"])
            
            # 3. 功能按鈕
            col_act1, col_act2 = st.columns([0.5, 0.5])
            with col_act1:
                # 單獨下載也要符合 PDF 資料夾輸出規定
                single_zip = io.BytesIO()
                with zipfile.ZipFile(single_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    folder_name = f"{ws['name']}_{ws['hkl']}_output/"
                    zf.writestr(f"{folder_name}POSCAR", zf.writestr(f"{folder_name}POSCAR", ws["poscar"]))
                    zf.writestr(f"{folder_name}slab_info.txt", ws["info_txt"])
                    zf.writestr(f"{folder_name}README_or_log.txt", ws["log_txt"])
                    
                html_link = create_download_link(single_zip.getvalue(), f"{ws['filename']}_output.zip", "📥 下載合規結構包 (ZIP)", is_zip=True)
                st.markdown(html_link, unsafe_allow_html=True)
            with col_act2:
                st.info("💡 提示：若要關閉此結構分頁，請點擊左側側邊欄「已載入結構清單」中該結構旁邊的❌。")
                
else:
    st.info("請在上方上傳一或多個金屬 POSCAR 檔案，並點擊「批次新增到工作區」。")
