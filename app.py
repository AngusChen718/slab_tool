import streamlit as st
import numpy as np
import ase
import ase.io
from ase.build import surface
import tempfile
import os
from io import StringIO
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Slab Builder Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 套用高質感學術大氣暗色主題
st.markdown("""
    <style>
        .main {
            background-color: #0f172a;
            color: #f8fafc;
        }
        .stSidebar {
            background-color: #1e293b !important;
            border-right: 1px solid #334155;
        }
        .css-1d391kg {
            background-color: #1e293b !important;
        }
        h1, h2, h3 {
            color: #38bdf8 !important;
            font-family: 'Noto Sans TC', sans-serif;
        }
        .metric-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #38bdf8;
        }
        .metric-label {
            font-size: 14px;
            color: #94a3b8;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("# ⚙️ 參數設定區")

# 1. 檔案上傳區（徹底拔除 type 限制，確保無副檔名的純 POSCAR 也能在所有作業系統順暢選取）
uploaded_file = st.sidebar.file_uploader(
    "1. 上傳 Bulk POSCAR / CIF",
    help="支援無副檔名 POSCAR 檔或標籤標準格式。瀏覽器不會強制進行副檔名過濾。"
)

st.sidebar.markdown("---")

# 2. 晶面指數與厚度設定
st.sidebar.markdown("### 2. 晶面定義 (Miller Index)")
col_h, col_k, col_l = st.sidebar.columns(3)
with col_h:
    h = int(st.sidebar.number_input("h", value=1, step=1))
with col_k:
    k = int(st.sidebar.number_input("k", value=1, step=1))
with col_l:
    l = int(st.sidebar.number_input("l", value=1, step=1))

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. 幾何規格設定")
layers = int(st.sidebar.number_input("Slab 原子層數 (Layers)", value=4, min_value=1, step=1))
vacuum = st.sidebar.slider("真空層厚度 (Vacuum, Å)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)

st.sidebar.markdown("### 4. 擴張表面積 (Supercell)")
col_sx, col_sy = st.sidebar.columns(2)
with col_sx:
    super_x = int(st.sidebar.number_input("X 倍數", value=3, min_value=1, step=1))
with col_sy:
    super_y = int(st.sidebar.number_input("Y 倍數", value=3, min_value=1, step=1))

st.sidebar.markdown("---")
# 4. Selective Dynamics 約束設定
st.sidebar.markdown("### 5. 表面計算優化設定")
fix_ratio = st.sidebar.slider("固定底部原子比例 (%)", min_value=0, max_value=100, value=40, step=5)
st.sidebar.info("💡 提示：鎖定底部原子以模擬塊材（Bulk）環境，能加速 VASP 收斂。")

def process_slab(bulk_atoms, h, k, l, layers, vacuum, super_x, super_y, fix_ratio):
    """
    執行表面切削流水線。
    為了解決「多層真空分離 (千層派 Bug)」，在此我們採用無真空幾何緊密疊加，
    最後一步才動態調整晶胞 Z 軸高度並施加置中真空。
    """
    # 1. 緊密切削：不傳入任何 vacuum 參數，確保切出的 slab 原子間距 100% 維持真實金屬鍵距離
    slab = surface(bulk_atoms, (h, k, l), layers=layers, vacuum=None)
    
    # 2. 超晶胞二維擴張：Z 軸保持 1 倍，避免重複堆疊
    if super_x > 1 or super_y > 1:
        slab = slab * (super_x, super_y, 1)
        
    # 3. 邊界碎屑物理補正：集體微調 Z 軸座標，避開 Z=0 的臨界折返面
    slab.positions[:, 2] += 1.0
    
    # 4. 全局真空置中加載：
    # 利用 ASE 的 center 函式。將 vacuum 設為設定值的一半，
    # 這樣 ASE 會將 Z 軸邊界自動擴增為「實際厚度 + 2 * (vacuum/2)」，實現 Z 軸朝上完美真空。
    slab.center(vacuum=vacuum / 2.0, axis=2)
    slab.wrap()
    
    # 5. 施加 Selective Dynamics 鎖定標記 (標註 T T T / F F F)
    # 根據 Z 座標由低到高排序，決定鎖定對象
    z_coords = slab.positions[:, 2]
    min_z, max_z = np.min(z_coords), np.max(z_coords)
    threshold = min_z + (max_z - min_z) * (fix_ratio / 100.0)
    
    # 建立 constraints 陣列
    constraints = []
    for pos in slab.positions:
        if pos[2] <= threshold:
            constraints.append([False, False, False]) # 固定 (F F F)
        else:
            constraints.append([True, True, True])    # 自由 (T T T)
            
    return slab, constraints

st.title("🧪 Slab Builder Pro <span style='font-size:16px; color:#94a3b8;'>v2.1 (Active Development Mode)</span>", unsafe_allow_html=True)
st.write("材料表面前處理與幾何合規體檢中繼站。已整合 macOS 無副檔名 POSCAR 直接上傳支援與 VESTA 級 3D 週期性渲染。")

if uploaded_file is not None:
    # 取得原始檔名與內容
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    # 建立臨時檔以便交給 ASE 解析
    with tempfile.NamedTemporaryFile(delete=False, suffix="_bulk") as temp:
        temp.write(file_bytes)
        temp_path = temp.name
        
    try:
        # 自動分流讀取
        content_preview = file_bytes.decode("utf-8", errors="ignore")
        if "data_" in content_preview or filename.lower().endswith(".cif"):
            bulk = ase.io.read(temp_path, format="cif")
            st.success(f"✅ CIF 檔案載入成功！(晶胞包含 {len(bulk)} 個原子)")
        else:
            # 預設為 VASP POSCAR
            bulk = ase.io.read(temp_path, format="vasp")
            st.success(f"✅ POSCAR 檔案載入成功！(晶胞包含 {len(bulk)} 個原子)")
            
        os.unlink(temp_path)
        
        # 執行核心切片計算
        slab, constraints = process_slab(bulk, h, k, l, layers, vacuum, super_x, super_y, fix_ratio)
        
        # 利用 ASE 原生寫入 CIF 至 StringIO，確保晶格參數、對稱性完整保留，從根源消滅 XYZ 碎裂 Bug
        cif_io = StringIO()
        ase.io.write(cif_io, slab, format="cif")
        cif_data = cif_io.getvalue()
        
        st.subheader("👀 互動式 3D 原子結構體檢 (VESTA-like Viewer)")
        st.write("已全面升級為 CIF 單元通訊協議，具備原生晶胞外框 (Cell Box) 與 PBC 週期性正確鍵結判定。")
        
        py3dmol_html = f"""
        <div id="viewer_container" style="height: 500px; width: 100%; background-color: #0b0f19; border-radius: 12px; border: 1px solid #334155; position: relative;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <script>
            $(document).ready(function() {{
                let element = $('#viewer_container');
                let config = {{ backgroundColor: '#0b0f19' }};
                let viewer = $3Dmol.createViewer(element, config);
                
                // 載入具備晶格矩陣的 CIF 格式資料
                viewer.addModel(`{cif_data}`, "cif");
                
                // 設定高擬真原子球與棒狀金屬鍵樣式
                viewer.setStyle({{}}, {{
                    sphere: {{ scale: 0.28, colorscheme: 'rasmol' }},
                    stick: {{ radius: 0.08, colorscheme: 'rasmol' }}
                }});
                
                // 自動繪製完美對齊的 12 條晶界框線 (Crystalline Box)
                viewer.addUnitCell();
                
                viewer.zoomTo();
                viewer.render();
            }});
        </script>
        """
        components.html(py3dmol_html, height=520)
        
        st.subheader("📊 結構體檢與合規摘要")
        
        # 嚴格的三維向量叉積表面積計算 (適配所有非正交、斜方及六方晶系)
        cell = slab.get_cell()
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        total_atoms = len(slab)
        fixed_count = sum([1 for c in constraints if c == [False, False, False]])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">總原子數 (Total Atoms)</div>
                    <div class="metric-value">{total_atoms}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">真表面積 (Surface Area)</div>
                    <div class="metric-value">{area:.4f} Å²</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">鎖定原子數 (Fixed / Ratio)</div>
                    <div class="metric-value">{fixed_count} / {fix_ratio}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">真空高度 (Z-Vacuum)</div>
                    <div class="metric-value">{cell[2][2]:.2f} Å</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.subheader("💾 匯出 VASP 計算檔案")
        
        # 建立具備 Selective Dynamics 的標準 VASP POSCAR 字串
        poscar_io = StringIO()
        # 利用 ASE 寫入為 VASP 格式
        ase.io.write(poscar_io, slab, format="vasp", vasp5=True, direct=True)
        poscar_str = poscar_io.getvalue()
        
        # 注入 Selective Dynamics 標記
        lines = poscar_str.split('\n')
        modified_poscar = []
        
        for idx, line in enumerate(lines):
            if idx == 7:
                modified_poscar.append(line)
                modified_poscar.append("Selective dynamics")
            elif idx >= 8 and len(line.strip().split()) >= 3:
                # 取得該原子對應的 constraint
                atom_idx = idx - 8
                if atom_idx < len(constraints):
                    c = constraints[atom_idx]
                    tag = "T  T  T" if c == [True, True, True] else "F  F  F"
                    # 將座標與約束標籤合併
                    parts = line.strip().split()
                    coords = f"  {float(parts[0]):.10f}  {float(parts[1]):.10f}  {float(parts[2]):.10f}"
                    modified_poscar.append(f"{coords}   {tag}")
                else:
                    modified_poscar.append(line)
            else:
                modified_poscar.append(line)
                
        final_poscar = '\n'.join(modified_poscar)
        
        # 下載按鈕
        st.download_button(
            label="📥 下載標準 POSCAR (含 Selective Dynamics)",
            data=final_poscar,
            file_name=f"POSCAR_{h}{k}{l}_{layers}L",
            mime="text/plain"
        )
        
        st.code(final_poscar[:1000] + "\n\n... (後續座標已省略) ...", language="text")

    except Exception as e:
        st.error(f"❌ 解析出錯：{str(e)}")
        st.info("請檢查上傳的檔案內容是否為標準的 POSCAR 或 CIF 格式。")

else:
    # 未上傳檔案時的導引畫面
    st.info("💡 請在側邊欄上傳您的 Bulk 結構。系統支援無副檔名的純 VASP POSCAR 以及 CIF 格式。")
    
    # 畫一個精緻的技術架構示意圖
    st.markdown("""
        <div style="background-color: #1e293b; padding: 30px; border-radius: 12px; border: 1px solid #334155; margin-top: 20px;">
            <h3>🛠️ Slab Builder Pro 工作原理</h3>
            <p>1. <strong>讀取與分流</strong>：自動辨識 CIF/POSCAR 輸入，排除系統副檔名鎖定阻礙。</p>
            <p>2. <strong>高精度切片</strong>：基於矩陣投影演算法切出緊密原子層，消除非物理真空隔離。</p>
            <p>3. <strong>動態 Z 軸重構</strong>：以原子層實際厚度為基準動態加載置中真空，防範週期性邊緣碎屑。</p>
            <p>4. <strong>約束寫入</strong>：依照比例自動於坐標後方寫入 <code style="color:#38bdf8;">F F F</code> 與 <code style="color:#38bdf8;">T T T</code> 第一性原理計算約束條件。</p>
        </div>
    """, unsafe_allow_html=True)
```
