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

# --- 🛠️ 晶體結構自動識別邏輯 ---
def crystal_structure_classifier(atoms):
    try:
        from ase.neighborlist import NeighborList
        cutoffs = [1.6] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        nl.update(atoms)
        cn_list = [len(nl.get_neighbors(i)[0]) for i in range(len(atoms))]
        avg_cn = np.mean(cn_list) if cn_list else 0
        
        if 11.5 <= avg_cn <= 12.5:
            cell = atoms.get_cell()
            angles = cell.lengths_and_angles()[3:]
            if any(np.isclose(a, 120, atol=5) for a in angles):
                return "HCP (六方最密堆積)"
            return "FCC (面心立方結構)"
        elif 7.5 <= avg_cn <= 8.5:
            return "BCC (體心立方結構)"
        elif 5.5 <= avg_cn <= 6.5:
            return "SC (簡單立方結構)"
        else:
            return "未知/複雜低對稱結構"
    except Exception:
        return "不確定 (請參考原始文獻)"

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
        /* 針對分頁標題內部的按鈕進行定位和美化 */
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
        
        /* 滑鼠懸停在「X」上時的顏色變紅 */
        div.stTabs [data-testid="stMarkdownContainer"] button:hover {
            background-color: #FFDDDD !important;
            border-radius: 4px;
            color: #FF4B4B !important;
        }
        
        /* 針對選中狀態的分頁標題，讓其內部的叉叉顏色深一點 */
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

# --- 主程式：上傳區 ---
uploaded_files = st.file_uploader("上傳金屬 POSCAR (支援多檔案同時拖曳)", key="uploader", accept_multiple_files=True)

if uploaded_files:
    if st.button("➕ 批次新增到工作區", type="primary"):
        for uploaded_file in uploaded_files:
            try:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                bulk = read_vasp(stringio)
                
                detected_lattice = crystal_structure_classifier(bulk)
                
                # 切割與放大
                slab = surface(bulk, (h, k, l), layers=layers, vacuum=vacuum)
                if super_x > 1 or super_y > 1: slab = slab * (super_x, super_y, 1)
                slab.wrap()
                
                # 結構體檢
                dist = neighbor_list('d', slab, cutoff=1.2)
                health_status = "✅ 正常" if len(dist) == 0 else "⚠️ 異常(原子重疊)"
                
                # Selective Dynamics 邏輯 (主線 A)
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
                    "filename": f"POSCAR_{uploaded_file.name.split('.')[0]}_{h}{k}{l}.vasp",
                    "poscar": poscar_data,
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
        # 在側邊欄清單也加入叉叉關閉功能 (與 Tabs 同步)
        col_side1, col_side2 = st.sidebar.columns([0.85, 0.15])
        label = f"[{ws['health']}] {ws['name']} - ({ws['hkl']})"
        is_selected = col_side1.checkbox(label, value=True, key=f"select_side_{idx}")
        if is_selected:
            selected_indices.append(idx)
        
        # 側邊欄的關閉按鈕
        if col_side2.button("❌", key=f"del_side_{idx}"):
            st.session_state.workspaces.pop(idx)
            st.rerun()
            
    st.sidebar.markdown(" ")
    if selected_indices:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx in selected_indices:
                ws = st.session_state.workspaces[idx]
                zip_file.writestr(ws["filename"], ws["poscar"])
        
        html_link = create_download_link(zip_buffer.getvalue(), "selected_slabs.zip", "📦 批次打包選中結構 (ZIP)", is_zip=True)
        st.sidebar.markdown(html_link, unsafe_allow_html=True)
        st.sidebar.markdown(" ")
    
    if st.sidebar.button("🗑️ 清空所有工作區", type="secondary", use_container_width=True):
        st.session_state.workspaces = []
        st.rerun()

# --- 主畫面：分頁管理（VESTA 高階渲染 + 整合叉叉關閉功能） ---
if st.session_state.workspaces:
    # 🛠️ 關鍵修復：手動建構帶有「X」關閉按鈕的分頁標題
    titles = []
    for idx, ws in enumerate(st.session_state.workspaces):
        hkl_label = ws['hkl']
        # 利用 Streamlit 的 st.button 直接渲染在 Tabs 標題中
        # 這裡用 col1 占滿空間來放置標題文字，col2 放置一個小按鈕
        titles.append(f"{ws['name']} - ({hkl_label}) ❌")

    # 渲染 Tabs
    tabs = st.tabs(titles)
    
    # 由於 `st.tabs` 並不支援按鈕點擊事件，我們需要利用一個技巧來監測「哪個分頁的叉叉被點擊了」。
    # 我們在側邊欄清單中已經整合了關閉功能，這是對 Tabs 功能的完美補充。
    # 為了讓 Tabs「看起來」像有叉叉按鈕，我們利用 HTML/CSS 在 Tabs 標題中渲染❌。
    # **重要說明：在目前的 Streamlit 架構下，直接點擊 Tabs 標題內的「X」文字並不能觸發後端事件**。
    # **因此，我保留了側邊欄清單中的❌按鈕作為「關閉此分頁」的實際操作方法，這能確保程式邏輯的穩定性**。
    # Tabs 標題中的「X」僅作為視覺指示。
    
    # 修改 Tabs 標題，移除視覺上的 "❌" 以免誤導 (因為原生的 tabs 不支援點擊事件)
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
            
            # 3. 功能按鈕 (移除原本的❌按鈕)
            col_act1, col_act2 = st.columns([0.5, 0.5])
            with col_act1:
                html_link = create_download_link(ws["poscar"], ws["filename"], "📥 單獨下載此 POSCAR")
                st.markdown(html_link, unsafe_allow_html=True)
            with col_act2:
                # 這裡原本有❌按鈕，現在我們將關閉功能完全整合到側邊欄清單中，以實現「叉叉」關閉分頁的效果。
                st.info("💡 提示：若要關閉此結構分頁，請點擊左側側邊欄「已載入結構清單」中該結構旁邊的❌。")
                
else:
    st.info("請在上方上傳一或多個金屬 POSCAR 檔案，並點擊「批次新增到工作區」。")