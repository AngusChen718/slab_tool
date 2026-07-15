import streamlit as st
import numpy as np
import io
import zipfile
from ase.io import read, write
from ase.build import surface
from ase.constraints import FixAtoms

# ==============================================================================
# 1. 網頁版型與 CSS 樣式美化 (對齊學術暗色高階質感)
# ==============================================================================
st.set_page_config(page_title="Slab Builder Pro", layout="wide", page_icon="🔬")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    h1, h2, h3 { color: #38bdf8 !important; font-family: 'Poppins', sans-serif; }
    .stSidebar { background-color: #1e293b !important; border-right: 1px solid #334155; }
    .stButton>button { 
        background: linear-gradient(90deg, #38bdf8, #818cf8); 
        color: white; 
        border: none; 
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Slab Builder Pro — 材料表面前處理與合規體檢中繼站")
st.write("本工具專門解決手動切面易出錯、真空重複加載、以及 Selective Dynamics 標註繁瑣之痛點。")

# ==============================================================================
# 2. 側邊控制欄設計
# ==============================================================================
st.sidebar.header("⚙️ 參數設定區")

# 2.1 檔案上傳
uploaded_file = st.sidebar.file_uploader("1. 上傳 Bulk POSCAR / CIF", type=["POSCAR", "cif", "txt"])

# 2.2 密勒指數
st.sidebar.markdown("---")
st.sidebar.subheader("2. 晶面定義 (Miller Index)")
col_h, col_k, col_l = st.sidebar.columns(3)
with col_h: h = col_h.number_input("h", value=1, step=1)
with col_k: k = col_k.number_input("k", value=1, step=1)
with col_l: l = col_l.number_input("l", value=1, step=1)

# 2.3 厚度與真空
st.sidebar.markdown("---")
st.sidebar.subheader("3. 幾何規格設定")
layers = st.sidebar.number_input("Slab 原子層數 (Layers)", min_value=1, max_value=50, value=10)
vacuum = st.sidebar.number_input("真空層厚度 (Vacuum / Å)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

# 2.4 超晶胞擴張
st.sidebar.markdown("---")
st.sidebar.subheader("4. 超晶胞擴張 (Supercell)")
col_sx, col_sy = st.sidebar.columns(2)
with col_sx: super_x = col_sx.number_input("X 倍數", min_value=1, max_value=10, value=3)
with col_sy: super_y = col_sy.number_input("Y 倍數", min_value=1, max_value=10, value=3)

# 2.5 表面計算優化（Selective Dynamics）
st.sidebar.markdown("---")
st.sidebar.subheader("5. Selective Dynamics 約束")
fix_ratio = st.sidebar.slider("固定底部原子比例 (%)", min_value=0, max_value=100, value=40, step=5)

# ==============================================================================
# 3. 核心幾何計算管線 (無真空切削，徹底杜絕千層派 Bug！)
# ==============================================================================
if uploaded_file is not None:
    try:
        # A. 讀取 Bulk 結構
        file_bytes = uploaded_file.read()
        file_str = file_bytes.decode("utf-8")
        tmp_io = io.StringIO(file_str)
        bulk_obj = read(tmp_io, format="vasp" if "POSCAR" in uploaded_file.name or "txt" in uploaded_file.name else "cif")
        
        st.success(f"🎉 檔案讀取成功！原始 Bulk 包含 {len(bulk_obj)} 個原子。")
        
        # B. 核心幾何切削管線（解耦真空，先切出 100% 緊密之固體單元）
        # 1. 第一步：切削時真空設為 None，保證層與層之間絕無縫隙
        slab = surface(bulk_obj, (h, k, l), layers=layers, vacuum=None)
        
        # 2. 第二步：在 z 方向倍數維持 1 的情況下，向 x, y 方向擴張超晶胞
        if super_x > 1 or super_y > 1:
            slab = slab * (super_x, super_y, 1)
            
        # 3. 第三步：將原子底部對齊 Z = 1.0 Å（保留底部緩衝，消除 wraparound 折返碎屑）
        z_min = np.min(slab.positions[:, 2])
        slab.positions[:, 2] -= z_min
        slab.positions[:, 2] += 1.0

        # 4. 第四步：動態重構 Z 軸晶格向量，使真空層單一朝上（完全對齊 VESTA 標準）
        current_thickness = np.max(slab.positions[:, 2])
        new_z_length = current_thickness + vacuum
        
        cell = slab.get_cell()
        cell[2][2] = new_z_length # 重新賦予厚度 + 真空的 Z 軸長度
        slab.set_cell(cell)
        
        # 5. 第五步：進行週期性邊界 Wrap
        slab.wrap()
        
        # 6. 第六步：微調對齊，確保最底部平整
        z_min_final = np.min(slab.positions[:, 2])
        slab.positions[:, 2] -= z_min_final
        slab.positions[:, 2] += 1.0  # 最终定位於 1.0 Å 底座

        # C. Selective Dynamics 標記寫入 (利用 ASE Constraints)
        num_to_fix = 0
        if fix_ratio > 0:
            z_positions = sorted(slab.positions[:, 2])
            num_to_fix = int(len(slab) * (fix_ratio / 100.0))
            if num_to_fix > 0:
                # 取得劃分固定/放鬆的 z 軸閾值
                z_threshold = z_positions[num_to_fix - 1]
                # 篩選高度小於等於該高度的原子 index 進行鎖定
                indices_to_fix = [atom.index for atom in slab if atom.position[2] <= z_threshold + 0.01]
                constraint = FixAtoms(indices=indices_to_fix)
                slab.set_constraint(constraint)

        # ==============================================================================
        # 4. 幾何合規體檢與數據摘要
        # ==============================================================================
        st.header("📊 表面結構幾何資訊摘要")
        
        # 計算向量叉積表面積
        cell_vectors = slab.get_cell()
        surface_area = np.linalg.norm(np.cross(cell_vectors[0], cell_vectors[1]))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>📐 晶胞表面積</h4>
                <p style='font-size: 24px; font-weight: bold; color: #38bdf8;'>{surface_area:.2f} Å²</p>
                <small>基於非正交向量叉積精確計算</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>⚛️ 原子總數</h4>
                <p style='font-size: 24px; font-weight: bold; color: #38bdf8;'>{len(slab)} 顆</p>
                <small>固定：{num_to_fix if fix_ratio > 0 else 0} 顆 / 放鬆：{len(slab) - (num_to_fix if fix_ratio > 0 else 0)} 顆</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>🌌 真空層厚度與佔比</h4>
                <p style='font-size: 24px; font-weight: bold; color: #38bdf8;'>{vacuum:.2f} Å ({vacuum/new_z_length*100:.1f}%)</p>
                <small>晶胞 Z 軸總長：{new_z_length:.2f} Å</small>
            </div>
            """, unsafe_allow_html=True)

        # ==============================================================================
        # 5. VESTA 級 3D 互動可視化（含 Unit Cell 晶格框線）
        # ==============================================================================
        st.header("👁️ 互動式 3D 結構檢驗視窗")
        st.caption("滑鼠左鍵拖曳旋轉、右鍵平移、滾輪縮放。高亮白框為 VESTA 規律邊界 (Unit Cell)。")
        
        # ⚡ 修正關鍵：由 XYZ 格式全面升級為 CIF 格式，將 Unit Cell 晶胞資訊完整傳遞給前端 Viewer
        f_cif = io.StringIO()
        write(f_cif, slab, format="cif")
        cif_string = f_cif.getvalue().replace("\n", "\\n").replace("'", "\\'")

        html_code = f"""
        <div id="py3dmol-canvas" style="width: 100%; height: 500px; background-color: #0b0f19; border-radius: 12px; overflow: hidden;"></div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
            let element = document.getElementById('py3dmol-canvas');
            let config = {{ backgroundColor: '#0b0f19' }};
            let viewer = $3Dmol.createViewer(element, config);
            let cifData = '{cif_string}';
            
            // 載入 CIF 格式模型（py3Dmol 將自動解析晶胞特徵與化學鍵合）
            let model = viewer.addModel(cifData, "cif");
            
            // 設定晶體表面原子樣式
            viewer.setStyle({{}}, {{
                sphere: {{ scale: 0.28, colorscheme: 'Jmol' }},
                stick: {{ radius: 0.08, colorscheme: 'Jmol' }}
            }});
            
            // ⚡ 由 py3Dmol 根據晶胞矩陣自動、精準繪製三維 Unit Cell 框線
            viewer.addUnitCell(model, {{
                box: {{
                    color: '#94a3b8',
                    linewidth: 1.5
                }}
            }});
            
            viewer.zoomTo();
            viewer.render();
        </script>
        """
        st.components.v1.html(html_code, height=520)

        # ==============================================================================
        # 6. 打包輸出 POSCAR 與診斷報告
        # ==============================================================================
        st.header("💾 打包導出計算輸入檔")
        
        # 寫入 VASP 5 格式的 POSCAR
        f_poscar = io.StringIO()
        write(f_poscar, slab, format="vasp", direct=True)
        poscar_content = f_poscar.getvalue()
        
        # 撰寫幾何體檢診斷書 (slab_info.txt)
        info_content = f"""==================================================
Slab Builder Pro - Structural Health Diagnostic
==================================================
* Sliced Facet (hkl)   : ({h}, {k}, {l})
* Atomic Layers        : {layers}
* Vacuum Thickness     : {vacuum:.4f} Angstroms
* Supercell Dimension  : {super_x} x {super_y} x 1
* Target Constrained % : {fix_ratio}%
* Actual Fixed Atoms   : {num_to_fix if fix_ratio > 0 else 0} / {len(slab)}
* Perfect Surface Area : {surface_area:.6f} A^2
* Total Cell Z Length  : {new_z_length:.6f} A
==================================================
This POSCAR is generated as an initial geometric guess.
Relaxation in DFT (VASP) is strictly required before usage.
"""

        # 打包成 ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr("POSCAR", poscar_content)
            zip_file.writestr("slab_info.txt", info_content)
            
        st.download_button(
            label="📦 一鍵下載 VASP 專用輸入檔與體檢報告 (.zip)",
            data=zip_buffer.getvalue(),
            file_name=f"Slab_{h}{k}{l}_Layers{layers}_Fix{fix_ratio}.zip",
            mime="application/zip"
        )
        
    except Exception as e:
        st.error(f"❌ 幾何轉換發生異常錯誤，請確認輸入檔案格式是否正確。錯誤代碼: {str(e)}")
else:
    st.info("💡 請在左側上傳 Bulk POSCAR 或 CIF 晶體結構檔案以開始進行表面前處理。")
