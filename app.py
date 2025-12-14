import streamlit as st
import pandas as pd
import time
from streamlit_option_menu import option_menu
import numpy as np
<<<<<<< HEAD
=======
import plotly.express as px
>>>>>>> repo-bim-dss/main

# --- ERROR HANDLING UNTUK MODUL CUSTOM ---
try:
    import preprocessing as pp
    import model_utils as mu
<<<<<<< HEAD
except ImportError as e:
    st.error(f"❌ Modul custom tidak ditemukan: {e}. Pastikan file 'preprocessing.py' dan 'model_utils.py' ada di folder yang sama.")
=======
    import database as db
except ImportError as e:
    st.error(f"❌ Modul custom tidak ditemukan: {e}. Pastikan file 'preprocessing.py', 'model_utils.py', dan 'database.py' ada di folder yang sama.")
>>>>>>> repo-bim-dss/main
    st.stop()

# =============================================================================
# 1. KONFIGURASI HALAMAN & SESSION STATE
# =============================================================================
st.set_page_config(
    page_title="MBA & ANN Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi Session State
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'data' not in st.session_state: st.session_state.data = None
if 'model' not in st.session_state: st.session_state.model = None
if 'association_rules' not in st.session_state: st.session_state.association_rules = None
if 'antecedents' not in st.session_state: st.session_state.antecedents = None
<<<<<<< HEAD
if 'key_col' not in st.session_state: st.session_state.key_col = None
if 'product_list_col' not in st.session_state: st.session_state.product_list_col = None
if 'demo_features' not in st.session_state: st.session_state.demo_features = None
=======
# RFM Analysis Session State
if 'rfm_data' not in st.session_state: st.session_state.rfm_data = None
if 'rfm_calculated' not in st.session_state: st.session_state.rfm_calculated = False
# Basket Configuration State
if 'basket_group_by' not in st.session_state: st.session_state.basket_group_by = 'BASKET_ID'
if 'basket_product_level' not in st.session_state: st.session_state.basket_product_level = 'COMMODITY_DESC'
# ANN Additional State
if 'feature_importance' not in st.session_state: st.session_state.feature_importance = None
if 'target_product' not in st.session_state: st.session_state.target_product = None

# Fixed column mappings (from database schema - original names)
KEY_COL = "household_key"
PRODUCT_LIST_COL = "product_list"
DAY_COL = "DAY"
DEMO_FEATURES = ["AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC", "HOMEOWNER_DESC", 
                "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC"]
CONTACT_COL = "phone_number"  # Customer contact info for campaigns


def get_active_demo_features():
    """Return the currently selected demographic features, preserving default order."""
    selected = st.session_state.get("selected_demo_features")
    if isinstance(selected, list):
        ordered = [feat for feat in DEMO_FEATURES if feat in selected]
        if ordered:
            return ordered
    return DEMO_FEATURES
>>>>>>> repo-bim-dss/main

# =============================================================================
# 2. GLOBAL CSS & STYLING (HYBRID THEME: DARK SIDEBAR - LIGHT CONTENT)
# =============================================================================
st.markdown("""
    <style>
        /* --- 1. MAIN CONTENT (LIGHT THEME) --- */
        .stApp { background-color: #eff2f6; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
        .stApp p, .stApp li, .stApp span, .stApp div, .stApp label { color: #31333F !important; }

        /* --- 2. SIDEBAR STYLING (WHITE THEME) --- */
        [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e0e0e0; }
        [data-testid="stSidebar"] * { color: #31333F !important; }
        .sidebar-title { font-size: 22px; font-weight: 800; color: #2e7bcf !important; text-align: center; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 2px solid #2e7bcf; }
        .sidebar-footer { text-align: center; font-size: 12px; color: #94a3b8 !important; margin-top: 50px; }

        /* --- 3. INPUT & UPLOADER --- */
        [data-testid="stFileUploader"] { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
        div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #31333F !important; border: 1px solid #d1d5db !important; }
        span[data-baseweb="tag"] { background-color: #e6f3ff !important; color: #2e7bcf !important; border: 1px solid #2e7bcf !important; font-weight: 600 !important; }

        /* --- 4. CARDS & METRICS --- */
        [data-testid="stMetric"] { background-color: #ffffff; padding: 15px 20px !important; border-radius: 12px; border: 1px solid #e2e8f0; border-left: 6px solid #2e7bcf; margin-right: 10px !important; margin-bottom: 10px !important; min-height: 110px !important; display: flex; flex-direction: column; justify-content: center; }
        [data-testid="stMetricValue"] { color: #2e7bcf !important; font-size: 26px !important; font-weight: 800 !important; }

        /* --- 5. HEADERS & BUTTONS --- */
        .main-header { font-size: 30px; font-weight: 800; color: #1a202c !important; margin-bottom: 20px !important; }
        .sub-header { font-size: 16px; color: #555 !important; background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #ffb703; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 25px !important; }
        div.stButton > button:first-child { background-color: #ffffff !important; border: 2px solid #2e7bcf !important; border-radius: 8px !important; padding: 0.6rem 1.2rem !important; transition: all 0.3s ease !important; }
        div.stButton > button:first-child:hover { background-color: #2e7bcf !important; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(46, 123, 207, 0.3) !important; }
        div.stButton > button:first-child:hover * { color: #ffffff !important; }
        div.stButton > button:first-child * { color: #2e7bcf !important; font-weight: 700 !important; font-size: 16px !important; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛒 Market Basket<br>& Neural Network</div>', unsafe_allow_html=True)
    
    selected_page = option_menu(
        menu_title=None,
<<<<<<< HEAD
        options=["Upload Data", "Association Rules", "ANN Training", "Prediction Results", "Business Insights"],
        icons=["cloud-upload", "diagram-3", "cpu", "graph-up-arrow", "lightbulb"],
=======
        options=["Database", "Association Rules", "RFM Analysis", "Product Affinity", "ANN Training", "Prediction Results", "Business Insights"],
        icons=["database", "diagram-3", "people", "heart", "cpu", "graph-up-arrow", "lightbulb"],
>>>>>>> repo-bim-dss/main
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#2e7bcf", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "5px", "color": "#475569", "font-weight": "500"},
            "nav-link-selected": {"background-color": "#e0f2fe", "color": "#0284c7", "font-weight": "bold", "border-left": "4px solid #0284c7"},
        }
    )
    
    st.markdown("---")
<<<<<<< HEAD
    if st.session_state.data_loaded:
        st.success(f"✅ Data Ready\n\nRows: {st.session_state.data.shape[0]}")
    else:
        st.info("ℹ️ Menunggu Data")
=======
    if db.database_exists():
        tx_count = db.get_transaction_count()
        cust_count = db.get_customer_count()
        st.success(f"✅ Database Ready\n\n🛒 {tx_count:,} Transaksi\n👥 {cust_count:,} Pelanggan")
    else:
        st.warning("⚠️ Database belum dibuat")
>>>>>>> repo-bim-dss/main
    st.markdown('<div class="sidebar-footer">© 2025 Project Dashboard</div>', unsafe_allow_html=True)

# =============================================================================
# 4. HALAMAN UTAMA (LOGIC)
# =============================================================================

<<<<<<< HEAD
# --- PAGE 1: UPLOAD DATA ---
if selected_page == "Upload Data":
    st.markdown('<div class="main-header">📂 Data Setup & Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mulai analisis Anda dengan mengunggah dataset transaksi dan memetakan variabel kunci.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload File CSV", type="csv", label_visibility="collapsed")

    if uploaded_file and not st.session_state.data_loaded:
        with st.spinner("🔄 Membaca dan memproses dataset..."):
            try:
                data = pp.load_and_preprocess_data(uploaded_file)
                st.session_state.data = data
                st.session_state.data_loaded = True
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Gagal memuat file: {e}")

    if st.session_state.data_loaded:
        df = st.session_state.data
        all_columns = df.columns.tolist()

        col1, col2, col3, col4 = st.columns(4, gap="large")
        col1.metric("Total Transaksi", f"{df.shape[0]:,}")
        col2.metric("Jumlah Fitur", f"{df.shape[1]}")
        col3.metric("Ukuran Memori", f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
        col4.metric("Status", "✅ Active")

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1], gap="large")

        # --- PERBAIKAN: Update nama kolom default ke Bahasa Indonesia ---
        def find_idx(cols, candidates):
            for c in candidates:
                if c in cols: return cols.index(c)
            return 0

        with c1:
            st.info("🛠️ **Mapping Kolom**")
            with st.container(border=True):
                st.caption("Tentukan kolom identitas & produk.")
                # Tambahkan 'ID Pelanggan' ke daftar pencarian
                key_col = st.selectbox("Customer ID / Transaksi", all_columns, 
                                     index=find_idx(all_columns, ['ID Pelanggan', 'household_key', 'user_id', 'basket_id']))
                # Tambahkan 'Keranjang Belanja' ke daftar pencarian
                product_list_col = st.selectbox("List Produk (Items)", all_columns, 
                                              index=find_idx(all_columns, ['Keranjang Belanja', 'product_list', 'items', 'products']))
                
                if df[product_list_col].dtype == 'object':
                    st.success("Format kolom produk valid.", icon="✔️")
                else:
                    st.warning("Kolom produk bukan string.", icon="⚠️")

        with c2:
            st.success("🤖 **Fitur AI / Demografi**")
            with st.container(border=True):
                st.caption("Pilih fitur untuk input Neural Network.")
                # Filter kolom ID agar tidak masuk ke training (PENTING untuk performa)
                exclude_keywords = ['ID', 'id', 'key', 'KEY', 'PX']
                exclude_cols = [c for c in all_columns if any(k in c for k in exclude_keywords)] + [key_col, product_list_col]
                
                avail = [c for c in all_columns if c not in exclude_cols]
                
                demo_features = st.multiselect("Pilih Variabel:", avail, default=avail[:5] if avail else None)
                
                if demo_features: st.caption(f"Model akan belajar dari **{len(demo_features)}** fitur.")
                else: st.caption("⚠️ Minimal pilih 1 fitur.")

        st.markdown("<br>", unsafe_allow_html=True)
        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Simpan Konfigurasi & Lanjut ➡️", use_container_width=True):
                if not demo_features:
                    st.toast("Harap pilih fitur demografi!", icon="🚫")
                else:
                    st.session_state.key_col = key_col
                    st.session_state.product_list_col = product_list_col
                    st.session_state.demo_features = demo_features
                    st.toast("Konfigurasi tersimpan!", icon="💾")
                    time.sleep(1)
        
        with st.expander("🔍 Lihat Sampel Data Mentah"):
            st.dataframe(df.head(10), use_container_width=True)

# --- PAGE 2: ASSOCIATION RULES ---
elif selected_page == "Association Rules":
    st.markdown('<div class="main-header">🔗 Association Rules Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Temukan pola pembelian bersamaan menggunakan algoritma FP-Growth.</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded or not st.session_state.product_list_col:
        st.warning("⚠️ Silakan upload data dan simpan konfigurasi di halaman pertama.")
    else:
        with st.container(border=True):
            col_param, col_act = st.columns([3, 1])
            with col_param:
                min_support_val = st.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001, format="%.3f")
            with col_act:
                st.markdown("<br>", unsafe_allow_html=True)
                run_arm = st.button("Jalankan Analisis", use_container_width=True)

        if run_arm:
            with st.spinner("⏳ Menjalankan FP-Growth..."):
                try:
                    df = st.session_state.data
                    p_col = st.session_state.product_list_col
                    
                    data_arm = pp.convert_product_list(df.copy(), p_col)
                    rules, antecedents = pp.run_association_rules(data_arm, p_col, min_support=min_support_val)
                    
                    st.session_state.association_rules = rules
                    st.session_state.antecedents = antecedents
                    st.success("✅ Selesai!")
                except Exception as e:
                    st.error(f"Error ARM: {e}")

        # ... (Kode tombol 'Jalankan Analisis' di atas tetap sama) ...

        # Hasil Table (Versi Ramah Pengguna)
        if st.session_state.association_rules is not None:
            rules = st.session_state.association_rules
            
            st.markdown("### 📊 Pola Belanja yang Ditemukan")
            st.success(f"Berhasil menemukan **{len(rules)}** pola kebiasaan pelanggan.")
            
            # --- KONVERSI KE BAHASA AWAM ---
            
            # 1. Ambil kolom yang relevan
            cols_to_show = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
            # Pastikan kolom ada sebelum diambil
            valid_cols = [c for c in cols_to_show if c in rules.columns]
            
            # 2. Buat copy dataframe khusus untuk tampilan (agar data asli tidak rusak)
            display_df = rules[valid_cols].copy()
            
            # 3. Ganti Nama Kolom Menjadi Kalimat yang Dimengerti
            rename_map = {
                'antecedents_str': 'Jika Pelanggan Membeli...',
                'consequents_str': '...Maka Cenderung Membeli',
                'support': 'Popularitas (%)',
                'confidence': 'Peluang Beli (%)',
                'lift': 'Kekuatan Hubungan (x Kali)'
            }
            display_df.rename(columns=rename_map, inplace=True)
            
            # 4. Format Angka (Opsional, tapi sangat disarankan agar cantik)
            # Ubah 0.15 jadi 15.0%
            if 'Popularitas (%)' in display_df.columns:
                display_df['Popularitas (%)'] = (display_df['Popularitas (%)'] * 100).round(2).astype(str) + '%'
                
            # Ubah 0.85 jadi 85.0%
            if 'Peluang Beli (%)' in display_df.columns:
                display_df['Peluang Beli (%)'] = (display_df['Peluang Beli (%)'] * 100).round(1).astype(str) + '%'
            
            # Ubah 2.5 jadi 2.5x
            if 'Kekuatan Hubungan (x Kali)' in display_df.columns:
                display_df['Kekuatan Hubungan (x Kali)'] = display_df['Kekuatan Hubungan (x Kali)'].round(2).astype(str) + 'x'

            # 5. Tampilkan Tabel
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # 6. Tambahkan "Kamus" untuk Edukasi User
            with st.expander("📚 Cara Membaca Tabel Ini (Klik untuk Info)"):
                st.markdown("""
                * **Jika Pelanggan Membeli...**: Barang pemicu yang sudah ada di keranjang.
                * **...Maka Cenderung Membeli**: Barang rekomendasi yang cocok ditawarkan.
                * **Popularitas**: Seberapa sering pasangan ini muncul (Semakin tinggi = Barang pasaran).
                * **Peluang Beli**: Seberapa yakin kita dia akan membeli barang rekomendasi tersebut (Misal 80% = Sangat Yakin).
                * **Kekuatan Hubungan**: 
                    * **> 1.0x**: Hubungan kuat (Cocok untuk Paket Bundling).
                    * **1.0x**: Kebetulan saja.
                """)
                
# --- PAGE 3: ANN TRAINING ---
elif selected_page == "ANN Training":
    st.markdown('<div class="main-header">🧠 Neural Network Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Latih model AI untuk memprediksi probabilitas pembelian barang target.</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded or not st.session_state.demo_features:
        st.warning("⚠️ Konfigurasi data belum lengkap. Silakan kembali ke halaman Upload.")
=======
# Helper function to load data from database with configuration
def load_data_from_db(group_by='BASKET_ID', product_level='COMMODITY_DESC', force_reload=False):
    """Load analysis data from database into session state."""
    if not db.database_exists():
        return False, "Database belum dibuat. Silakan buat database di halaman Database."
    
    # Check if we need to reload (config changed or first load)
    config_changed = (st.session_state.basket_group_by != group_by or 
                     st.session_state.basket_product_level != product_level)
    
    if not st.session_state.data_loaded or config_changed or force_reload:
        df, err = db.get_analysis_data(group_by=group_by, product_level=product_level)
        if err:
            return False, f"Gagal memuat data: {err}"
        if df.empty:
            return False, "Database kosong. Silakan muat data di halaman Database."
        
        st.session_state.data = df
        st.session_state.data_loaded = True
        st.session_state.basket_group_by = group_by
        st.session_state.basket_product_level = product_level
    
    return True, None

# --- PAGE 1: ASSOCIATION RULES ---
if selected_page == "Association Rules":
    st.markdown('<div class="main-header">🔗 Association Rules Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Temukan pola pembelian bersamaan menggunakan algoritma FP-Growth.</div>', unsafe_allow_html=True)

    if not db.database_exists():
        st.warning("⚠️ Database belum dibuat. Silakan buat database di halaman Database.")
    else:
        # === TABS FOR SETTINGS AND ANALYSIS ===
        tab_settings, tab_analysis, tab_results = st.tabs(["⚙️ Pengaturan Basket", "🔬 Analisis", "📊 Hasil"])
        
        # === TAB 1: SETTINGS ===
        with tab_settings:
            st.markdown("### 🛒 Konfigurasi Pembentukan Keranjang Belanja")
            st.markdown("Tentukan metodologi pengelompokan data transaksional menjadi unit analisis 'keranjang belanja' untuk identifikasi pola asosiasi pembelian.")
            
            basket_options = db.get_basket_options()
            
            # --- SECTION 1: DATA SOURCE ---
            st.markdown("---")
            st.markdown("#### 1️⃣ Sumber Data")
            
            col_src1, col_src2 = st.columns(2)
            with col_src1:
                st.markdown("**📋 Tabel yang Digunakan:**")
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0284c7;">
                    <b>🔗 JOIN dari 3 tabel:</b><br>
                    • <code>transactions</code> - Data transaksi penjualan<br>
                    • <code>products</code> - Master produk & kategori<br>
                    • <code>customers</code> - Demografi pelanggan
                </div>
                """, unsafe_allow_html=True)
            
            with col_src2:
                st.markdown("**📊 Statistik Data:**")
                tx_count = db.get_transaction_count()
                cust_count = db.get_customer_count()
                prod_count = db.get_product_count()
                
                st.markdown(f"""
                <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 4px solid #22c55e;">
                    • 🛒 <b>{tx_count:,}</b> Transaksi (BASKET_ID)<br>
                    • 👥 <b>{cust_count:,}</b> Pelanggan (household_key)<br>
                    • 📦 <b>{prod_count:,}</b> Produk unik
                </div>
                """, unsafe_allow_html=True)
            
            # --- SECTION 2: BASKET GROUPING ---
            st.markdown("---")
            st.markdown("#### 2️⃣ Pengelompokan Basket (Baris = 1 Keranjang)")
            st.caption("Definisikan kriteria agregasi produk untuk membentuk satu unit transaksi dalam analisis market basket. Setiap baris merepresentasikan satu keranjang belanja yang independen.")
            
            col_grp1, col_grp2 = st.columns([1, 1])
            
            with col_grp1:
                group_by = st.radio(
                    "**Kelompokkan berdasarkan:**",
                    options=list(basket_options['group_by'].keys()),
                    format_func=lambda x: basket_options['group_by'][x],
                    index=0 if st.session_state.basket_group_by == 'BASKET_ID' else 1,
                    key="cfg_group_by"
                )
            
            with col_grp2:
                if group_by == 'BASKET_ID':
                    st.info("""
                    **Per Transaksi (BASKET_ID)**
                    - Setiap kunjungan belanja = 1 keranjang
                    - Cocok untuk: *"Apa yang dibeli bersamaan dalam satu kali belanja?"*
                    """)
                else:
                    st.info("""
                    **Per Pelanggan (household_key)**
                    - Semua pembelian pelanggan digabung = 1 keranjang
                    - Cocok untuk: *"Apa yang biasa dibeli oleh pelanggan ini?"*
                    """)
            
            # --- SECTION 3: PRODUCT LEVEL ---
            st.markdown("---")
            st.markdown("#### 3️⃣ Level Granularitas Produk")
            st.caption("Tentukan tingkat hierarki taksonomi produk yang akan digunakan dalam analisis. Level granularitas tinggi menghasilkan kategori umum, sedangkan level rendah memberikan spesifikasi hingga brand.")
            
            col_prod1, col_prod2 = st.columns([1, 1])
            
            with col_prod1:
                product_level = st.radio(
                    "**Level produk:**",
                    options=list(basket_options['product_level'].keys()),
                    format_func=lambda x: basket_options['product_level'][x],
                    index=list(basket_options['product_level'].keys()).index(st.session_state.basket_product_level),
                    key="cfg_product_level"
                )
            
            with col_prod2:
                st.markdown("**📌 Contoh nilai:**")
                sample_df, _ = db.get_product_level_sample(product_level, limit=10)
                if sample_df is not None and not sample_df.empty:
                    sample_values = sample_df[product_level].tolist()
                    for val in sample_values[:6]:
                        st.caption(f"• `{val}`")
            
            # --- SECTION 4: PREVIEW & APPLY ---
            st.markdown("---")
            st.markdown("#### 4️⃣ Preview & Terapkan Konfigurasi")
            
            with st.expander("🔍 Lihat Query SQL yang akan digunakan"):
                preview_sql = f"""SELECT t.household_key, t.BASKET_ID, t.DAY,
    GROUP_CONCAT(DISTINCT p.{product_level}) as product_list,
    c.AGE_DESC, c.INCOME_DESC, ...
FROM transactions t
JOIN customers c ON t.household_key = c.household_key
JOIN products p ON t.PRODUCT_ID = p.PRODUCT_ID
GROUP BY t.household_key, t.{group_by}
ORDER BY t.household_key, t.DAY"""
                st.code(preview_sql, language="sql")
            
            config_changed = (st.session_state.basket_group_by != group_by or 
                            st.session_state.basket_product_level != product_level)
            
            if config_changed:
                st.warning("⚠️ Konfigurasi telah diubah. Klik tombol di bawah untuk menerapkan.")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("✅ Terapkan Konfigurasi & Muat Data", type="primary", use_container_width=True):
                    st.session_state.basket_group_by = group_by
                    st.session_state.basket_product_level = product_level
                    st.session_state.data_loaded = False
                    st.session_state.association_rules = None
                    st.rerun()
            
            st.markdown("---")
            st.markdown("**🎯 Konfigurasi Aktif:**")
            st.success(f"Pengelompokan: `{st.session_state.basket_group_by}` | Level Produk: `{st.session_state.basket_product_level}`")
        
        # === TAB 2: ANALYSIS ===
        with tab_analysis:
            st.markdown("### 🔬 Jalankan Analisis FP-Growth")
            
            data_ok, err_msg = load_data_from_db(
                group_by=st.session_state.basket_group_by,
                product_level=st.session_state.basket_product_level
            )
            
            if not data_ok:
                st.warning(f"⚠️ {err_msg}")
            else:
                rows, cols = st.session_state.data.shape
                st.info(
                    f"📊 **Data Aktif:** {rows:,} baris × {cols} kolom | Kelompok: `{st.session_state.basket_group_by}` | Level: `{st.session_state.basket_product_level}`"
                )
                st.caption(f"Ukuran dataset sebelum FP-Growth: `{rows:,} x {cols}` (baris × kolom)")
                
                with st.expander("👀 Preview Data (5 baris pertama)"):
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
                
                st.markdown("---")
                
                with st.container(border=True):
                    st.markdown("**⚙️ Parameter FP-Growth**")
                    col_param, col_info = st.columns([2, 1])
                    
                    with col_param:
                        # Sampling option for large datasets
                        total_rows = len(st.session_state.data)
                        if total_rows > 20000:
                            st.warning(f"⚠️ Dataset besar ({total_rows:,} transaksi). Sampling direkomendasikan untuk performa cepat.")
                            use_sampling = st.checkbox("🚀 Gunakan Sampling (lebih cepat)", value=True)
                            if use_sampling:
                                sample_size = st.slider("Jumlah sampel transaksi", 10000, min(total_rows, 50000), 20000, 5000,
                                    help="20K sampel biasanya cukup untuk pola yang representatif dengan waktu ~10 detik.")
                            else:
                                sample_size = total_rows
                        else:
                            use_sampling = False
                            sample_size = total_rows
                        
                        min_support_val = st.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001, format="%.3f",
                            help="Persentase minimum kemunculan itemset.")
                        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05, format="%.2f",
                            help="Minimum kepercayaan aturan.")
                        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.1, 0.1, format="%.1f",
                            help="Minimum kekuatan hubungan.")
                    
                    with col_info:
                        st.markdown("""
                        **📖 Panduan:**
                        - Support rendah = pola langka
                        - Confidence = kepercayaan
                        - Lift > 1 = korelasi positif
                        """)
                
                col_run1, col_run2, col_run3 = st.columns([1, 2, 1])
                with col_run2:
                    run_arm = st.button("🚀 Jalankan FP-Growth", use_container_width=True, type="primary")

                if run_arm:
                    with st.spinner("⏳ Menjalankan FP-Growth..."):
                        try:
                            df = st.session_state.data
                            p_col = PRODUCT_LIST_COL
                            
                            # Apply sampling if enabled
                            if use_sampling and sample_size < len(df):
                                df_sample = df.sample(n=sample_size, random_state=42)
                                st.info(f"📊 Menggunakan {sample_size:,} sampel dari {len(df):,} transaksi")
                            else:
                                df_sample = df
                            
                            data_arm = pp.convert_product_list(df_sample.copy(), p_col)
                            rules, antecedents = pp.run_association_rules(
                                data_arm, p_col, 
                                min_support=min_support_val,
                                min_confidence=min_confidence,
                                min_lift=min_lift
                            )
                            
                            st.session_state.association_rules = rules
                            st.session_state.antecedents = antecedents
                            
                            if rules is not None and not rules.empty:
                                st.success(f"✅ Ditemukan **{len(rules)}** pola. Lihat di tab **📊 Hasil**.")
                            else:
                                st.warning("⚠️ Tidak ditemukan pola. Coba turunkan parameter.")
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        # === TAB 3: RESULTS ===
        with tab_results:
            st.markdown("### 📊 Hasil Analisis - Pola Belanja")
            
            if st.session_state.association_rules is None or (hasattr(st.session_state.association_rules, 'empty') and st.session_state.association_rules.empty):
                st.info("ℹ️ Belum ada hasil. Jalankan analisis di tab **🔬 Analisis**.")
            else:
                rules = st.session_state.association_rules
                st.success(f"✅ Ditemukan **{len(rules)}** pola kebiasaan pelanggan.")
                
                cols_to_show = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
                valid_cols = [c for c in cols_to_show if c in rules.columns]
                display_df = rules[valid_cols].copy()
                
                rename_map = {
                    'antecedents_str': 'Jika Membeli...',
                    'consequents_str': '...Maka Membeli',
                    'support': 'Popularitas (%)',
                    'confidence': 'Peluang (%)',
                    'lift': 'Kekuatan (x)'
                }
                display_df.rename(columns=rename_map, inplace=True)
                
                if 'Popularitas (%)' in display_df.columns:
                    display_df['Popularitas (%)'] = (display_df['Popularitas (%)'] * 100).round(2).astype(str) + '%'
                if 'Peluang (%)' in display_df.columns:
                    display_df['Peluang (%)'] = (display_df['Peluang (%)'] * 100).round(1).astype(str) + '%'
                if 'Kekuatan (x)' in display_df.columns:
                    display_df['Kekuatan (x)'] = display_df['Kekuatan (x)'].round(2).astype(str) + 'x'

                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                csv = rules.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "association_rules.csv", "text/csv")
                
                with st.expander("📚 Cara Membaca Tabel"):
                    st.markdown("""
                    * **Jika Membeli...**: Barang pemicu di keranjang
                    * **Maka Membeli**: Rekomendasi bundling
                    * **Popularitas**: Seberapa sering muncul
                    * **Peluang**: Kepercayaan prediksi
                    * **Kekuatan**: > 1x = hubungan kuat
                    """)

# --- PAGE 2: RFM ANALYSIS ---
elif selected_page == "RFM Analysis":
    st.markdown('<div class="main-header">👥 Customer Value Segmentation (RFM)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Implementasi model segmentasi pelanggan berbasis tiga dimensi metrik: Recency (interval temporal sejak transaksi terakhir), Frequency (intensitas aktivitas pembelian), dan Monetary (nilai agregat transaksi) untuk klasifikasi nilai pelanggan.</div>', unsafe_allow_html=True)

    # Load data from database
    data_ok, err_msg = load_data_from_db(
        group_by=st.session_state.basket_group_by,
        product_level=st.session_state.basket_product_level
    )
    
    if not data_ok:
        st.warning(f"⚠️ {err_msg}")
    else:
        # Parameter RFM
        with st.container(border=True):
            st.markdown("**⚙️ Konfigurasi RFM**")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"""RFM akan dihitung menggunakan:
                - **Customer ID:** `{KEY_COL}`
                - **Kolom Waktu:** `{DAY_COL}`
                - **Basket:** `{PRODUCT_LIST_COL}` (jumlah item = Monetary proxy)
                """)
            with col2:
                run_rfm = st.button("🚀 Hitung RFM", use_container_width=True, type="primary")

        # Jalankan RFM Calculation
        if run_rfm:
            with st.spinner("⏳ Menghitung skor RFM untuk setiap pelanggan..."):
                try:
                    rfm_result = pp.calculate_rfm(
                        st.session_state.data,
                        KEY_COL,
                        DAY_COL,
                        PRODUCT_LIST_COL
                    )
                    st.session_state.rfm_data = rfm_result
                    st.session_state.rfm_calculated = True
                    st.success(f"✅ RFM berhasil dihitung untuk {len(rfm_result)} pelanggan!")
                except Exception as e:
                    st.error(f"Gagal menghitung RFM: {e}")

        # Tampilkan Hasil RFM
        if st.session_state.rfm_calculated and st.session_state.rfm_data is not None:
            rfm = st.session_state.rfm_data
            recommendations = pp.get_segment_recommendations()
            
            # --- EXECUTIVE METRICS ---
            st.markdown("### 📊 Ringkasan Pelanggan")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Pelanggan", f"{len(rfm):,}")
            m2.metric("Avg Recency", f"{rfm['Recency'].mean():.0f} hari")
            m3.metric("Avg Frequency", f"{rfm['Frequency'].mean():.0f} transaksi")
            m4.metric("Avg Monetary", f"{rfm['Monetary'].mean():,.0f} items")

            # --- SEGMENT DISTRIBUTION ---
            st.markdown("### 🎯 Distribusi Segmen Pelanggan")
            col_chart, col_table = st.columns([1, 1], gap="large")
            
            with col_chart:
                segment_counts = rfm['Segment'].value_counts().reset_index()
                segment_counts.columns = ['Segment', 'Jumlah']
                
                # Sort by priority
                segment_order = [rec for rec in recommendations.keys()]
                segment_counts['sort_order'] = segment_counts['Segment'].apply(
                    lambda x: segment_order.index(x) if x in segment_order else 99
                )
                segment_counts = segment_counts.sort_values('sort_order').drop('sort_order', axis=1)
                
                st.bar_chart(segment_counts.set_index('Segment'), color="#2e7bcf", height=350)

            with col_table:
                st.markdown("**Aksi yang Direkomendasikan:**")
                for segment in segment_counts['Segment'].tolist():
                    if segment in recommendations:
                        rec = recommendations[segment]
                        count = segment_counts[segment_counts['Segment'] == segment]['Jumlah'].values[0]
                        st.markdown(f"""
                        <div style="background-color: white; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {rec['color']};">
                            <b>{segment}</b> ({count} orang)<br>
                            <small style="color: #666;">{rec['action']}</small>
                        </div>
                        """, unsafe_allow_html=True)

            # --- RFM DETAIL TABLE ---
            st.markdown("### 📋 Detail Skor RFM per Pelanggan")
            
            # Filter by segment
            segment_filter = st.multiselect(
                "Filter Segmen:",
                options=rfm['Segment'].unique().tolist(),
                default=None
            )
            
            display_rfm = rfm.copy()
            if segment_filter:
                display_rfm = display_rfm[display_rfm['Segment'].isin(segment_filter)]
            
            # Tampilkan kolom yang relevan
            display_cols = [KEY_COL, 'Recency', 'Frequency', 'Monetary', 
                          'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Total_Score', 'Segment']
            display_cols = [c for c in display_cols if c in display_rfm.columns]
            
            st.dataframe(
                display_rfm[display_cols].sort_values('Total_Score', ascending=False),
                use_container_width=True,
                hide_index=True
            )

            # --- DOWNLOAD BUTTON ---
            st.markdown("---")
            col_dl1, col_dl2, _ = st.columns([1, 1, 2])
            with col_dl1:
                csv_full = rfm.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Semua Data RFM",
                    csv_full,
                    "rfm_all_customers.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col_dl2:
                # Download hanya segmen prioritas (Champions, Loyal, At Risk)
                priority_segments = ['🏆 Champions', '💎 Loyal Customers', '🚨 At Risk', '🔥 Can\'t Lose Them']
                priority_df = rfm[rfm['Segment'].isin(priority_segments)]
                csv_priority = priority_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Prioritas Tinggi",
                    csv_priority,
                    "rfm_priority_customers.csv",
                    "text/csv",
                    use_container_width=True
                )

            # --- SEGMENT EXPLANATION ---
            with st.expander("📚 Penjelasan Segmen RFM (Klik untuk Info)"):
                st.markdown("""
                | Segmen | Deskripsi | Prioritas |
                |--------|-----------|----------|
                | 🏆 Champions | Pelanggan terbaik, baru belanja, sering, dan banyak | Tinggi |
                | 💎 Loyal Customers | Pelanggan setia dengan nilai konsisten | Tinggi |
                | 🌟 Potential Loyalist | Pelanggan baru dengan potensi loyalitas | Sedang |
                | 🆕 New Customers | Pelanggan baru (1-2x transaksi) | Sedang |
                | ⚠️ Need Attention | Performa mulai menurun | Sedang |
                | 😴 About to Sleep | Mulai jarang belanja | Waspada |
                | 🚨 At Risk | Dulunya aktif, sekarang jarang | Kritis |
                | ❄️ Hibernating | Sudah lama tidak belanja | Rendah |
                | 🔥 Can't Lose Them | Pernah terbaik, sekarang hilang | Kritis |
                """)

# --- PAGE 3: PRODUCT AFFINITY ---
elif selected_page == "Product Affinity":
    st.markdown('<div class="main-header">Demographic-Based Product Affinity</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analisis preferensi produk berbasis segmentasi demografis untuk mengidentifikasi deviasi konsumsi antar kohor pelanggan.</div>', unsafe_allow_html=True)
    
    # Import required functions
    from database import (
        get_demographic_options, 
        get_product_affinity_by_demographic,
        get_top_products_by_segment,
        get_demographic_distribution,
        get_segment_comparison
    )
    
    # Configuration sidebar
    col_config1, col_config2 = st.columns(2)
    
    demo_options = get_demographic_options()
    
    with col_config1:
        selected_demo = st.selectbox(
            "📊 Pilih Dimensi Demografis",
            options=list(demo_options.keys()),
            format_func=lambda x: f"{demo_options[x]} ({x})"
        )
    
    with col_config2:
        product_level = st.selectbox(
            "📦 Level Produk",
            options=['DEPARTMENT', 'COMMODITY_DESC', 'SUB_COMMODITY_DESC', 'BRAND'],
            format_func=lambda x: {
                'DEPARTMENT': 'Department (Level Tinggi)',
                'COMMODITY_DESC': 'Commodity (Level Menengah)',
                'SUB_COMMODITY_DESC': 'Sub-Commodity (Detail)',
                'BRAND': 'Brand'
            }.get(x, x)
        )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview Demografis", "🎯 Affinity Analysis", "📊 Segment Comparison"])
    
    # --- TAB 1: DEMOGRAPHIC OVERVIEW ---
    with tab1:
        st.subheader(f"📊 Distribusi {demo_options.get(selected_demo, selected_demo)}")
        
        dist_df, dist_err = get_demographic_distribution(selected_demo)
        
        if dist_err:
            st.error(f"Error loading distribution: {dist_err}")
        elif dist_df is not None and not dist_df.empty:
            # Summary metrics
            total_customers = dist_df['customers'].sum()
            total_sales = dist_df['total_sales'].sum()
            total_transactions = dist_df['transactions'].sum()
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Pelanggan", f"{total_customers:,}")
            with m2:
                st.metric("Total Transaksi", f"{total_transactions:,}")
            with m3:
                st.metric("Total Penjualan", f"${total_sales:,.2f}")
            with m4:
                st.metric("Segmen Unik", len(dist_df))
            
            st.markdown("---")
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("**👥 Distribusi Pelanggan per Segmen**")
                fig_cust = px.pie(
                    dist_df, 
                    values='customers', 
                    names='segment',
                    title=f'Customer Distribution by {demo_options.get(selected_demo, selected_demo)}',
                    hole=0.4
                )
                fig_cust.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_cust, use_container_width=True)
            
            with col_chart2:
                st.markdown("**💰 Kontribusi Penjualan per Segmen**")
                fig_sales = px.bar(
                    dist_df.sort_values('total_sales', ascending=True),
                    x='total_sales',
                    y='segment',
                    orientation='h',
                    title=f'Sales by {demo_options.get(selected_demo, selected_demo)}',
                    color='total_sales',
                    color_continuous_scale='Greens'
                )
                fig_sales.update_layout(showlegend=False)
                st.plotly_chart(fig_sales, use_container_width=True)
            
            # Detailed table
            st.markdown("**📋 Detail Statistik per Segmen**")
            display_dist = dist_df.copy()
            display_dist.columns = ['Segmen', 'Pelanggan', 'Transaksi', 'Total Sales', 'Avg Spend/Customer']
            display_dist['% Pelanggan'] = (display_dist['Pelanggan'] / total_customers * 100).round(2)
            display_dist['% Sales'] = (display_dist['Total Sales'] / total_sales * 100).round(2)
            st.dataframe(display_dist, use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada data distribusi demografis yang ditemukan.")
    
    # --- TAB 2: AFFINITY ANALYSIS ---
    with tab2:
        st.subheader(f"🎯 Product Affinity Index - {demo_options.get(selected_demo, selected_demo)}")
        
        st.info("""
        **Interpretasi Affinity Index:**
        - **Index > 1.5**: Afinitas SANGAT TINGGI - probabilitas pembelian segmen ini secara signifikan melebihi baseline populasi keseluruhan
        - **Index 1.1 - 1.5**: Afinitas TINGGI - terdapat kecenderungan positif terhadap produk dengan deviasi di atas rata-rata populasi
        - **Index 0.9 - 1.1**: Afinitas NETRAL - distribusi preferensi konvergen dengan karakteristik populasi umum
        - **Index 0.5 - 0.9**: Afinitas RENDAH - terdapat resistensi atau ketidaktertarikan relatif terhadap kategori produk
        - **Index < 0.5**: Afinitas SANGAT RENDAH - probabilitas pembelian mendekati nilai minimal, mengindikasikan ketidaksesuaian dengan preferensi segmen
        """)
        
        affinity_df, aff_err = get_product_affinity_by_demographic(selected_demo, product_level)
        
        if aff_err:
            st.error(f"Error loading affinity data: {aff_err}")
        elif affinity_df is not None and not affinity_df.empty:
            # Segment selector
            segments = affinity_df['segment'].unique().tolist()
            selected_segment = st.selectbox(
                "🔍 Pilih Segmen untuk Analisis Detail",
                options=segments,
                key="affinity_segment_selector"
            )
            
            # Filter for selected segment
            segment_data = affinity_df[affinity_df['segment'] == selected_segment].copy()
            
            # Display top affinity products
            col_aff1, col_aff2 = st.columns(2)
            
            with col_aff1:
                st.markdown(f"**🔥 Top 10 Produk dengan Affinity Tertinggi untuk '{selected_segment}'**")
                top_affinity = segment_data.nlargest(10, 'affinity_index')
                
                fig_top = px.bar(
                    top_affinity,
                    x='affinity_index',
                    y='product',
                    orientation='h',
                    title=f'Highest Affinity Products',
                    color='affinity_index',
                    color_continuous_scale='RdYlGn'
                )
                fig_top.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                                 annotation_text="Baseline (1.0)")
                fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top, use_container_width=True)
            
            with col_aff2:
                st.markdown(f"**❄️ Produk dengan Affinity Terendah untuk '{selected_segment}'**")
                low_affinity = segment_data.nsmallest(10, 'affinity_index')
                
                fig_low = px.bar(
                    low_affinity,
                    x='affinity_index',
                    y='product',
                    orientation='h',
                    title=f'Lowest Affinity Products',
                    color='affinity_index',
                    color_continuous_scale='RdYlGn'
                )
                fig_low.add_vline(x=1.0, line_dash="dash", line_color="gray",
                                 annotation_text="Baseline (1.0)")
                fig_low.update_layout(yaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_low, use_container_width=True)
            
            # Full data table
            st.markdown("**📋 Data Lengkap Affinity Analysis**")
            display_aff = segment_data[['product', 'segment_buyers', 'segment_customers', 
                                        'segment_penetration', 'overall_penetration', 
                                        'affinity_index', 'total_sales']].copy()
            display_aff.columns = ['Produk', 'Pembeli Segmen', 'Total Segmen', 
                                   'Penetrasi Segmen (%)', 'Penetrasi Overall (%)',
                                   'Affinity Index', 'Total Sales']
            
            # Color code affinity
            def highlight_affinity(val):
                if val > 1.5:
                    return 'background-color: #28a745; color: white'
                elif val > 1.1:
                    return 'background-color: #90EE90'
                elif val < 0.5:
                    return 'background-color: #dc3545; color: white'
                elif val < 0.9:
                    return 'background-color: #ffcccb'
                return ''
            
            styled_df = display_aff.style.applymap(highlight_affinity, subset=['Affinity Index'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Business recommendations
            st.markdown("---")
            st.subheader("💡 Rekomendasi Bisnis")
            
            high_aff = segment_data[segment_data['affinity_index'] > 1.3].nlargest(3, 'total_sales')
            low_aff = segment_data[segment_data['affinity_index'] < 0.7].nlargest(3, 'total_sales')
            
            rec1, rec2 = st.columns(2)
            with rec1:
                st.success(f"**✅ Produk Andalan untuk '{selected_segment}':**")
                if not high_aff.empty:
                    for _, row in high_aff.iterrows():
                        st.write(f"• **{row['product']}** (Index: {row['affinity_index']:.2f})")
                    st.caption("➡️ Fokuskan promosi produk ini ke segmen ini")
                else:
                    st.write("Tidak ada produk dengan affinity tinggi yang signifikan")
            
            with rec2:
                st.warning(f"**⚠️ Potensi Pengembangan untuk '{selected_segment}':**")
                if not low_aff.empty:
                    for _, row in low_aff.iterrows():
                        st.write(f"• **{row['product']}** (Index: {row['affinity_index']:.2f})")
                    st.caption("➡️ Peluang cross-selling dengan strategi khusus")
                else:
                    st.write("Segmen ini sudah memiliki preferensi merata")
        else:
            st.warning("Tidak ada data affinity yang ditemukan.")
    
    # --- TAB 3: SEGMENT COMPARISON ---
    with tab3:
        st.subheader(f"📊 Perbandingan Antar Segmen - {demo_options.get(selected_demo, selected_demo)}")
        
        comparison_df, comp_err = get_segment_comparison(selected_demo, product_level)
        
        if comp_err:
            st.error(f"Error loading comparison data: {comp_err}")
        elif comparison_df is not None and not comparison_df.empty:
            # Create pivot table for heatmap
            pivot_df = comparison_df.pivot_table(
                values='pct_of_segment_sales',
                index='product',
                columns='segment',
                fill_value=0
            )
            
            # Filter top products by total sales
            top_n_products = st.slider("Jumlah produk teratas untuk ditampilkan", 5, 30, 15)
            
            product_totals = comparison_df.groupby('product')['sales'].sum().nlargest(top_n_products)
            pivot_filtered = pivot_df.loc[product_totals.index]
            
            # Heatmap
            st.markdown("**🔥 Heatmap: % Kontribusi Produk per Segmen**")
            
            fig_heatmap = px.imshow(
                pivot_filtered.values,
                labels=dict(x="Segmen", y="Produk", color="% of Sales"),
                x=pivot_filtered.columns.tolist(),
                y=pivot_filtered.index.tolist(),
                color_continuous_scale='YlOrRd',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Stacked bar chart
            st.markdown("**📊 Komposisi Penjualan per Segmen**")
            
            top_products_list = product_totals.index.tolist()
            stacked_data = comparison_df[comparison_df['product'].isin(top_products_list)]
            
            fig_stacked = px.bar(
                stacked_data,
                x='segment',
                y='pct_of_segment_sales',
                color='product',
                title=f'Product Mix by {demo_options.get(selected_demo, selected_demo)}',
                labels={'pct_of_segment_sales': '% of Segment Sales', 'segment': 'Segment'}
            )
            fig_stacked.update_layout(barmode='stack', height=500)
            st.plotly_chart(fig_stacked, use_container_width=True)
            
            # Download pivot table
            st.markdown("**📥 Download Data Perbandingan**")
            csv = pivot_filtered.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"segment_comparison_{selected_demo}_{product_level}.csv",
                mime="text/csv"
            )
            
            # Key insights
            st.markdown("---")
            st.subheader("🔍 Key Insights")
            
            # Find most differentiated products
            if len(pivot_filtered.columns) > 1:
                variance_by_product = pivot_filtered.var(axis=1).sort_values(ascending=False)
                
                st.markdown("**Produk dengan Preferensi Paling Bervariasi Antar Segmen:**")
                for i, (product, var) in enumerate(variance_by_product.head(5).items()):
                    max_seg = pivot_filtered.loc[product].idxmax()
                    max_val = pivot_filtered.loc[product].max()
                    min_seg = pivot_filtered.loc[product].idxmin()
                    min_val = pivot_filtered.loc[product].min()
                    
                    st.write(f"{i+1}. **{product}**: Tertinggi di '{max_seg}' ({max_val:.1f}%), "
                            f"Terendah di '{min_seg}' ({min_val:.1f}%)")
        else:
            st.warning("Tidak ada data perbandingan yang ditemukan.")
    
    # Footer with methodology
    st.markdown("---")
    with st.expander("📚 Metodologi Product Affinity Analysis"):
        st.markdown("""
        ### Definisi Product Affinity Index
        
        Product Affinity Index merupakan parameter kuantitatif yang mengevaluasi intensitas preferensi suatu segmen demografis 
        terhadap kategori produk tertentu dengan baseline acuan berupa distribusi preferensi populasi keseluruhan.
        
        **Formula Matematis:**
        ```
        Affinity Index = (Penetrasi di Segmen) / (Penetrasi Overall)
        
        dimana:
        - Penetrasi Segmen = Proporsi pelanggan dalam segmen yang melakukan akuisisi produk (%)
        - Penetrasi Overall = Proporsi total pelanggan dalam populasi yang melakukan akuisisi produk (%)
        ```
        
        **Interpretasi Statistik:**
        - **Index = 1.0**: Distribusi pembelian segmen identik dengan populasi, tidak terdapat bias preferensi
        - **Index > 1.0**: Terdapat bias positif, menandakan overrepresentation perilaku pembelian pada segmen
        - **Index < 1.0**: Terdapat bias negatif, menandakan underrepresentation perilaku pembelian pada segmen
        
        ### Aplikasi Manajerial:
        1. **Targeted Marketing**: Optimalkan alokasi anggaran promosi ke segmen dengan afinitas tinggi untuk memaksimalkan ROI
        2. **Product Development**: Rancang pengembangan produk berbasis insight preferensi segmen dengan pembuktian empiris
        3. **Cross-selling**: Identifikasi gap penetrasi guna merancang strategi ekspansi kategori pada segmen potensial
        4. **Inventory Planning**: Sinkronkan komposisi inventori dengan profil demografis wilayah guna meminimalkan stockout maupun overstock
        """)
                
# --- PAGE 4: ANN TRAINING ---
elif selected_page == "ANN Training":
    st.markdown('<div class="main-header">🧠 Neural Network Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Implementasi dan training model Artificial Neural Network untuk memprediksi probabilitas pembelian produk target serta mengekstraksi karakteristik profil pembeli potensial melalui analisis feature importance.</div>', unsafe_allow_html=True)

    # Load data from database
    data_ok, err_msg = load_data_from_db(
        group_by=st.session_state.basket_group_by,
        product_level=st.session_state.basket_product_level
    )
    
    if not data_ok:
        st.warning(f"⚠️ {err_msg}")
>>>>>>> repo-bim-dss/main
    else:
        c1, c2 = st.columns([1, 1], gap="large")
        
        with c1:
            st.info("🎯 **Target Prediksi**")
            with st.container(border=True):
                ant_opts = [""] + (st.session_state.antecedents if st.session_state.antecedents else [])
                sel_ant = st.selectbox("Pilih dari pola populer (Opsional):", ant_opts)
                
                target_str = st.text_input("Atau ketik Produk Target (koma separator):", value=sel_ant if sel_ant else "")
                target_list = set(p.strip().upper() for p in target_str.split(',') if p)

<<<<<<< HEAD
=======
        default_demo_selection = st.session_state.get("selected_demo_features")
        if not isinstance(default_demo_selection, list) or not default_demo_selection:
            default_demo_selection = DEMO_FEATURES
        else:
            default_demo_selection = [feat for feat in DEMO_FEATURES if feat in default_demo_selection]
            if not default_demo_selection:
                default_demo_selection = DEMO_FEATURES

>>>>>>> repo-bim-dss/main
        with c2:
            st.success("⚙️ **Parameter Model**")
            with st.container(border=True):
                resample = st.selectbox("Penanganan Data Tidak Seimbang:", 
                                      ['oversampling', 'undersampling'], 
<<<<<<< HEAD
                                      format_func=lambda x: "SMOTE (Oversampling) - Recommended" if x == 'oversampling' else "Random Undersampling")
=======
                                      format_func=lambda x: "SMOTE (Oversampling)" if x == 'oversampling' else "Random Undersampling  - Recommended")
                selected_demo_features = st.multiselect(
                    "Pilih fitur demografis untuk dimasukkan ke ANN:",
                    options=DEMO_FEATURES,
                    default=default_demo_selection,
                    help="Hilangkan kolom yang kurang relevan agar model fokus pada faktor demografis yang paling penting."
                )
                st.session_state.selected_demo_features = selected_demo_features
>>>>>>> repo-bim-dss/main

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Mulai Training Model", type="primary", use_container_width=True):
            if not target_list:
                st.error("Tentukan produk target terlebih dahulu!")
<<<<<<< HEAD
=======
            elif not st.session_state.get("selected_demo_features"):
                st.error("Pilih minimal satu fitur demografis sebelum melatih model.")
>>>>>>> repo-bim-dss/main
            else:
                with st.spinner("🤖 Sedang melatih model (Pre-processing > Encoding > Training)..."):
                    try:
                        df = st.session_state.data
<<<<<<< HEAD
                        p_col = st.session_state.product_list_col
                        d_feats = st.session_state.demo_features
=======
                        p_col = PRODUCT_LIST_COL
                        d_feats = get_active_demo_features()
>>>>>>> repo-bim-dss/main
                        
                        data_ann = pp.convert_product_list(df.copy(), p_col)
                        data_target = pp.create_target_variable(data_ann, p_col, target_list)
                        data_enc = pp.encode_features(data_target, d_feats)

                        y_full = data_enc['PX']
                        orig_cols = set(data_target.columns)
                        final_cols = set(data_enc.columns)
                        X_full = data_enc[list(final_cols - orig_cols)].copy() 

                        st.session_state.X_full = X_full
<<<<<<< HEAD
                        st.session_state.full_keys = data_enc[[st.session_state.key_col, p_col, 'PX']]
=======
                        st.session_state.full_keys = data_enc[[KEY_COL, p_col, 'PX']]
                        st.session_state.target_product = ", ".join(target_list)  # Store target product
>>>>>>> repo-bim-dss/main

                        X_train, y_train, X_test, y_test = mu.split_and_resample(X_full, y_full, method=resample)
                        
                        with st.expander("Lihat Distribusi Data Training"):
                            st.write("Target Distribution (Train):", y_train.value_counts())
                        
                        model = mu.train_ann_model(X_train, y_train)
                        st.session_state.model = model
                        st.session_state.eval_metrics = mu.generate_evaluation_metrics(model, X_test, y_test)
                        
<<<<<<< HEAD
=======
                        # Calculate Feature Importance using permutation importance
                        from sklearn.inspection import permutation_importance
                        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                        
                        # Store feature importance
                        feature_imp_df = pd.DataFrame({
                            'feature': X_full.columns,
                            'importance': perm_importance.importances_mean,
                            'std': perm_importance.importances_std
                        }).sort_values('importance', ascending=False)
                        st.session_state.feature_importance = feature_imp_df
                        
>>>>>>> repo-bim-dss/main
                        probs, preds = mu.get_predictions(model, X_full)
                        res_df = st.session_state.full_keys.copy()
                        res_df['Probability'] = probs
                        res_df['Prediction'] = preds
<<<<<<< HEAD
=======
                        
                        # Merge demographic data and contact info for buyer persona analysis
                        demo_cols = [KEY_COL] + d_feats
                        if CONTACT_COL in df.columns:
                            demo_cols.append(CONTACT_COL)
                        demo_data = df[demo_cols].drop_duplicates(subset=[KEY_COL])
                        res_df = res_df.merge(demo_data, on=KEY_COL, how='left')
                        
>>>>>>> repo-bim-dss/main
                        st.session_state.prediction_results = res_df
                        
                        st.success("✅ Training Selesai! Lihat hasil detail di menu 'Prediction Results'.")
                        
                    except Exception as e:
                        st.error(f"Gagal training: {e}")

<<<<<<< HEAD
# --- PAGE 4: RESULTS ---
elif selected_page == "Prediction Results":
    st.markdown('<div class="main-header">📈 Model Evaluation & Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analisis performa model dan hasil prediksi pada seluruh pelanggan.</div>', unsafe_allow_html=True)
=======
# --- PAGE 5: RESULTS ---
elif selected_page == "Prediction Results":
    st.markdown('<div class="main-header">📈 Actionable Marketing Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Buyer Persona, Feature Importance, dan Target List untuk kampanye produk.</div>', unsafe_allow_html=True)
>>>>>>> repo-bim-dss/main

    if not st.session_state.model:
        st.info("⚠️ Model belum dilatih. Silakan ke menu ANN Training.")
    else:
        evals = st.session_state.eval_metrics
<<<<<<< HEAD
        
        m1, m2 = st.columns(2)
        m1.metric("AUC-ROC Score", f"{evals['auc']:.4f}")
        m2.metric("Accuracy (Test Set)", f"{evals['report']['accuracy']:.4f}" if 'accuracy' in evals['report'] else "N/A")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("**Confusion Matrix**")
                st.pyplot(evals['cm_plot'], use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown("**ROC Curve**")
                st.pyplot(evals['roc_plot'], use_container_width=True)

        st.markdown("### 📋 Hasil Prediksi Pelanggan")
        res_df = st.session_state.prediction_results
        st.dataframe(res_df, use_container_width=True)

        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil (CSV)",
            data=csv,
            file_name="hasil_prediksi_ann.csv",
            mime="text/csv",
            type="primary"
        )
        
# --- PAGE 5: BUSINESS INSIGHTS ---
elif selected_page == "Business Insights":
    st.markdown('<div class="main-header">💡 Strategic Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Gabungan kekuatan pola MBA dan presisi prediksi ANN untuk strategi pemasaran.</div>', unsafe_allow_html=True)

    if not st.session_state.association_rules is not None:
        st.warning("⚠️ Harap jalankan 'Association Rules' terlebih dahulu.")
    elif not 'prediction_results' in st.session_state:
        st.warning("⚠️ Harap jalankan 'ANN Training' terlebih dahulu.")
    else:
        # --- 0. PREPARE DATA (DEDUPLIKASI DI AWAL) ---
        # Kita buat dataframe level 'Customer' (Unik), bukan Transaksi
        raw_df = st.session_state.prediction_results
        key_col = st.session_state.key_col
        
        # Ambil probabilitas tertinggi per customer
        unique_customers_df = raw_df.sort_values(by='Probability', ascending=False).drop_duplicates(subset=[key_col], keep='first')
        
        # --- 1. AI SMART CONCLUSION ---
        st.markdown("### 🧠 AI Smart Conclusion")
        
        # Ambil Top Profile dari data UNIK
        hot_leads_unique = unique_customers_df[unique_customers_df['Probability'] > 0.75]
        
        insight_box_content = []
        
        # Insight Demografi (Dari user unik)
        original_df = st.session_state.data
        if not hot_leads_unique.empty and st.session_state.demo_features:
            # Kita perlu merge data demografi asli ke list unik ini
            # Karena prediction_results mungkin tidak menyimpan semua kolom demografi
            hot_ids = hot_leads_unique[key_col].unique()
            profile_data = original_df[original_df[key_col].isin(hot_ids)]
            
            dom_traits = []
            for feature in st.session_state.demo_features[:3]:
                if feature in profile_data.columns:
                    try:
                        top_val = profile_data[feature].mode()[0]
                        clean_feat = feature.replace('_', ' ').title()
                        dom_traits.append(f"<b>{clean_feat} {top_val}</b>")
                    except:
                        pass # Handle jika mode kosong
            
            if dom_traits:
                traits_str = ", ".join(dom_traits)
                insight_box_content.append(f"🎯 <b>Profil Target Utama:</b> Pelanggan prioritas memiliki profil dominan {traits_str}.")

        # Insight Pola Belanja
        rules = st.session_state.association_rules
        if rules is not None and not rules.empty:
            top_rule = rules.sort_values(by='lift', ascending=False).iloc[0]
            ant = top_rule['antecedents_str']
            con = top_rule['consequents_str']
            lift = top_rule['lift']
            insight_box_content.append(f"🛒 <b>Pola Pemicu:</b> Promosi produk <b>{ant}</b> sangat efektif memicu pembelian <b>{con}</b> (Lift: {lift:.1f}x).")

        insight_box_content.append(f"🚀 <b>Strategi:</b> Fokuskan budget marketing pada {len(hot_leads_unique)} pelanggan unik di bawah ini.")

        html_content = "<br><br>".join(insight_box_content)
        st.markdown(f"""
        <div style="background-color: #f0f9ff; border-left: 6px solid #2e7bcf; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); font-size: 16px; color: #334155; margin-bottom: 30px;">
            <h3 style="margin-top:0; color:#2e7bcf;">✨ Kesimpulan Strategis</h3>
            {html_content}
        </div>
        """, unsafe_allow_html=True)

        # --- 2. EXECUTIVE METRICS (DATA UNIK) ---
        auc_score = st.session_state.eval_metrics['auc']
        
        # Hitung jumlah Hot Leads UNIK (Orang), bukan Transaksi
        hot_leads_count = len(hot_leads_unique)
        total_unique_customers = len(unique_customers_df)
        hot_leads_pct = (hot_leads_count / total_unique_customers * 100) if total_unique_customers > 0 else 0
            
        max_lift = rules['lift'].max() if not rules.empty else 0

        # Status Logic
        if auc_score > 0.75 and hot_leads_count > 10:
            verdict = "STRATEGI SANGAT POTENSIAL (GO)"
            status_color = "green"
        elif auc_score < 0.6:
            verdict = "MODEL KURANG AKURAT (NO GO)"
            status_color = "red"
        else:
            verdict = "CUKUP POTENSIAL (CAUTION)"
            status_color = "orange"

        m1, m2, m3 = st.columns(3)
        m1.metric("Kualitas Prediksi (AUC)", f"{auc_score:.2f}", delta="Akurasi Model")
        # Metric ini sekarang menampilkan jumlah ORANG
        m2.metric("Hot Leads (Pelanggan Unik)", f"{hot_leads_count} Org", f"{hot_leads_pct:.1f}% dari Total Pelanggan")
        m3.metric("Kekuatan Pola (Max Lift)", f"{max_lift:.2f}x", delta="Daya Tarik Produk")
        
        st.markdown(f"<div style='text-align:center; color:{status_color}; font-weight:bold; margin-bottom:20px;'>STATUS: {verdict}</div>", unsafe_allow_html=True)

        # --- 3. VISUALIZATIONS ---
        c1, c2 = st.columns([1, 1], gap="large")
        
        with c1:
            st.markdown("### 📊 Pemicu Produk (Top Rules)")
            top_rules = rules.sort_values(by="lift", ascending=False).head(8)
            chart_data = top_rules[['antecedents_str', 'lift']].copy()
            chart_data['Rule'] = chart_data['antecedents_str'] + " ➡️ " + top_rules['consequents_str']
            st.bar_chart(chart_data.set_index('Rule')['lift'], color="#2e7bcf")

        with c2:
            st.markdown("### 👥 Segmentasi Pelanggan (Unik)")
            # Hitung segmentasi berdasarkan data UNIK
            conditions = [
                (unique_customers_df['Probability'] >= 0.8),
                (unique_customers_df['Probability'] >= 0.5) & (unique_customers_df['Probability'] < 0.8),
                (unique_customers_df['Probability'] < 0.5)
            ]
            choices = ['🔥 Hot Leads', '☁️ Warm', '❄️ Cold']
            unique_customers_df['Segment'] = np.select(conditions, choices, default='Unknown')
            
            segment_counts = unique_customers_df['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Jumlah Orang']
            st.bar_chart(segment_counts.set_index('Segment'), color="#ffb703")

        # --- 4. ACTION PLAN TABLE ---
        st.markdown("---")
        st.markdown("### 🚀 Daftar Target Prioritas (Top 50 Unik)")
        
        # Ambil Top 50 dari data unik yang sudah disiapkan
        top_targets = unique_customers_df.head(50)
        
        # Merge kembali demografi untuk ditampilkan di tabel
        desired_cols = [key_col, 'Probability', 'Segment']
        if st.session_state.demo_features:
            # Ambil data demografi unik
            demo_data = original_df[[key_col] + st.session_state.demo_features].drop_duplicates(subset=[key_col])
            # Merge left ke top_targets
            top_targets_final = top_targets.merge(demo_data, on=key_col, how='left')
            
            # Update kolom yang mau ditampilkan
            available_demo = [c for c in st.session_state.demo_features if c in top_targets_final.columns]
            desired_cols += available_demo[:3]
        else:
            top_targets_final = top_targets

        valid_cols = [c for c in desired_cols if c in top_targets_final.columns]
        
        st.dataframe(
            top_targets_final[valid_cols].style.background_gradient(subset=['Probability'], cmap='Blues'),
            use_container_width=True
        )
        
        csv = top_targets_final[valid_cols].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download List Marketing", csv, "target_marketing.csv", "text/csv", type="primary")
=======
        res_df = st.session_state.prediction_results
        target_product = st.session_state.get('target_product', 'Unknown')
        active_demo_features = get_active_demo_features()
        
        # Header with target product
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px;">
            <h3 style="margin: 0; color: white;">🎯 Target Produk: {target_product}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Hasil analisis prediksi pembelian dan profil pembeli potensial</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab_persona, tab_importance, tab_targets, tab_model = st.tabs([
            "👤 Buyer Persona", "📊 Feature Importance", "🎯 Target List", "📈 Model Performance"
        ])
        
        # =================================================================
        # TAB 1: BUYER PERSONA
        # =================================================================
        with tab_persona:
            st.markdown("## 👤 Buyer Persona Generator")
            st.markdown("Profil demografis pelanggan yang **kemungkinan besar membeli** produk target.")
            
            # Get True Positives (High probability customers who we should target)
            unique_pred = res_df.drop_duplicates(subset=[KEY_COL])
            true_positives = unique_pred[unique_pred['Probability'] > 0.6]  # Predicted buyers
            actual_buyers = unique_pred[unique_pred['PX'] == 1]  # Actual historical buyers
            
            col_tp, col_ab = st.columns(2)
            
            with col_tp:
                st.markdown("### 🔮 Predicted Buyers (Probability > 60%)")
                st.metric("Jumlah", f"{len(true_positives):,} pelanggan")
                
                if not true_positives.empty:
                    st.markdown("**Profil Dominan:**")
                    persona_data = []
                    
                    for feat in active_demo_features:
                        if feat in true_positives.columns:
                            value_counts = true_positives[feat].value_counts()
                            if not value_counts.empty:
                                top_val = value_counts.index[0]
                                top_pct = value_counts.iloc[0] / len(true_positives) * 100
                                persona_data.append({
                                    'Demografis': feat.replace('_', ' ').title(),
                                    'Nilai Dominan': top_val,
                                    'Persentase': f"{top_pct:.1f}%"
                                })
                    
                    if persona_data:
                        persona_df = pd.DataFrame(persona_data)
                        st.dataframe(persona_df, use_container_width=True, hide_index=True)
            
            with col_ab:
                st.markdown("### 📚 Historical Buyers (Actual)")
                st.metric("Jumlah", f"{len(actual_buyers):,} pelanggan")
                
                if not actual_buyers.empty:
                    st.markdown("**Profil Dominan:**")
                    actual_persona = []
                    
                    for feat in active_demo_features:
                        if feat in actual_buyers.columns:
                            value_counts = actual_buyers[feat].value_counts()
                            if not value_counts.empty:
                                top_val = value_counts.index[0]
                                top_pct = value_counts.iloc[0] / len(actual_buyers) * 100
                                actual_persona.append({
                                    'Demografis': feat.replace('_', ' ').title(),
                                    'Nilai Dominan': top_val,
                                    'Persentase': f"{top_pct:.1f}%"
                                })
                    
                    if actual_persona:
                        actual_df = pd.DataFrame(actual_persona)
                        st.dataframe(actual_df, use_container_width=True, hide_index=True)
            
            # Visual Persona Card
            st.markdown("---")
            st.markdown("### 🎨 Visual Buyer Persona Card")
            
            if not true_positives.empty:
                # Build persona summary
                persona_traits = {}
                for feat in active_demo_features[:5]:
                    if feat in true_positives.columns:
                        top_val = true_positives[feat].mode()
                        if not top_val.empty:
                            persona_traits[feat] = top_val.iloc[0]
                
                age = persona_traits.get('AGE_DESC', 'N/A')
                income = persona_traits.get('INCOME_DESC', 'N/A')
                marital = persona_traits.get('MARITAL_STATUS_CODE', 'N/A')
                homeowner = persona_traits.get('HOMEOWNER_DESC', 'N/A')
                kids = persona_traits.get('KID_CATEGORY_DESC', 'N/A')
                
                st.markdown(f"""
                <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e2e8f0; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 60px;">👤</div>
                        <h2 style="margin: 10px 0; color: #1e40af;">Target Buyer Persona</h2>
                        <p style="color: #64748b;">untuk produk <b>{target_product}</b></p>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
                        <div style="background: #dbeafe; padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px;">🎂</div>
                            <div style="font-weight: bold; color: #1e40af;">Usia</div>
                            <div style="color: #334155;">{age}</div>
                        </div>
                        <div style="background: #dcfce7; padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px;">💰</div>
                            <div style="font-weight: bold; color: #166534;">Pendapatan</div>
                            <div style="color: #334155;">{income}</div>
                        </div>
                        <div style="background: #fef3c7; padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px;">💍</div>
                            <div style="font-weight: bold; color: #92400e;">Status</div>
                            <div style="color: #334155;">{marital}</div>
                        </div>
                        <div style="background: #fce7f3; padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px;">🏠</div>
                            <div style="font-weight: bold; color: #9d174d;">Rumah</div>
                            <div style="color: #334155;">{homeowner}</div>
                        </div>
                        <div style="background: #e0e7ff; padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px;">👶</div>
                            <div style="font-weight: bold; color: #4338ca;">Anak</div>
                            <div style="color: #334155;">{kids}</div>
                        </div>
                        <div style="background: #f1f5f9; padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px;">📊</div>
                            <div style="font-weight: bold; color: #475569;">Total Target</div>
                            <div style="color: #334155;">{len(true_positives):,} orang</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Marketing Platform Recommendations
                st.markdown("---")
                st.markdown("### 📢 Rekomendasi Platform Marketing")
                
                platform_cols = st.columns(3)
                with platform_cols[0]:
                    st.markdown(f"""
                    <div style="background: #1877f2; color: white; padding: 15px; border-radius: 10px;">
                        <h4 style="color: white;">📘 Facebook Ads</h4>
                        <small>
                        • Age: {age}<br>
                        • Income: {income}<br>
                        • Interests: Retail, Shopping<br>
                        • Household: {homeowner}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with platform_cols[1]:
                    st.markdown(f"""
                    <div style="background: #ea4335; color: white; padding: 15px; border-radius: 10px;">
                        <h4 style="color: white;">🔍 Google Ads</h4>
                        <small>
                        • Keywords: {target_product}<br>
                        • Demographics: {age}<br>
                        • Household Income: {income}<br>
                        • Parental: {kids}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with platform_cols[2]:
                    st.markdown(f"""
                    <div style="background: #25d366; color: white; padding: 15px; border-radius: 10px;">
                        <h4 style="color: white;">📱 WhatsApp/Email</h4>
                        <small>
                        • Target: {len(true_positives)} contacts<br>
                        • Personalization: {age}<br>
                        • Offer: Product promo<br>
                        • Timing: Weekend
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # =================================================================
        # TAB 2: FEATURE IMPORTANCE
        # =================================================================
        with tab_importance:
            st.markdown("## 📊 Feature Importance - Apa yang Mempengaruhi Pembelian?")
            st.markdown("Faktor demografis mana yang paling menentukan apakah pelanggan akan membeli produk target.")
            
            if 'feature_importance' in st.session_state and st.session_state.feature_importance is not None:
                feat_imp = st.session_state.feature_importance
                
                # Top 15 features
                top_features = feat_imp.head(15)
                
                # Bar chart
                fig_imp = px.bar(
                    top_features.sort_values('importance', ascending=True),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig_imp.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Group by demographic category
                st.markdown("### 📋 Importance by Demographic Category")
                
                # Parse feature names to get original demographic
                demo_importance = {}
                for _, row in feat_imp.iterrows():
                    for demo in active_demo_features:
                        if row['feature'].startswith(demo):
                            if demo not in demo_importance:
                                demo_importance[demo] = 0
                            demo_importance[demo] += max(0, row['importance'])
                            break
                
                if demo_importance:
                    demo_imp_df = pd.DataFrame([
                        {'Demografis': k.replace('_', ' ').title(), 'Total Importance': v}
                        for k, v in demo_importance.items()
                    ]).sort_values('Total Importance', ascending=False)
                    
                    fig_demo = px.bar(
                        demo_imp_df,
                        x='Demografis',
                        y='Total Importance',
                        title='Aggregated Importance by Demographic',
                        color='Total Importance',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig_demo, use_container_width=True)
                    
                    # Interpretation
                    top_demo = demo_imp_df.iloc[0]['Demografis']
                    st.success(f"""
                    **💡 Key Insight:**  
                    **{top_demo}** adalah faktor demografis yang paling berpengaruh dalam memprediksi pembelian **{target_product}**.
                    Fokuskan segmentasi dan targeting berdasarkan faktor ini.
                    """)
            else:
                st.info("Feature importance belum tersedia. Jalankan training ulang untuk melihat feature importance.")
        
        # =================================================================
        # TAB 3: TARGET LIST
        # =================================================================
        with tab_targets:
            st.markdown("## 🎯 Campaign Target List")
            st.markdown("Daftar pelanggan yang harus ditarget untuk kampanye produk.")
            
            unique_pred = res_df.drop_duplicates(subset=[KEY_COL])
            
            # Filter options
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                prob_threshold = st.slider("Minimum Probability", 0.0, 1.0, 0.6, 0.05)
            
            with col_filter2:
                exclude_buyers = st.checkbox("Exclude existing buyers (PX=1)", value=True)
            
            # Apply filters
            target_list_df = unique_pred[unique_pred['Probability'] >= prob_threshold]
            if exclude_buyers:
                target_list_df = target_list_df[target_list_df['PX'] == 0]
            
            # Summary
            st.markdown("---")
            total_targets = len(target_list_df)
            
            sum_cols = st.columns(4)
            sum_cols[0].metric("🎯 Total Targets", f"{total_targets:,}")
            sum_cols[1].metric("🔥 Hot (>75%)", len(target_list_df[target_list_df['Probability'] > 0.75]))
            sum_cols[2].metric("☁️ Warm (60-75%)", len(target_list_df[(target_list_df['Probability'] >= 0.6) & (target_list_df['Probability'] <= 0.75)]))
            sum_cols[3].metric("📊 Avg Probability", f"{target_list_df['Probability'].mean()*100:.1f}%")
            
            # Demographic breakdown of targets
            st.markdown("### 📊 Demographic Breakdown of Targets")
            
            demo_breakdown_cols = st.columns(3)
            
            for i, feat in enumerate(active_demo_features[:3]):
                if feat in target_list_df.columns:
                    with demo_breakdown_cols[i]:
                        breakdown = target_list_df[feat].value_counts().head(5)
                        st.markdown(f"**{feat.replace('_', ' ').title()}**")
                        for val, count in breakdown.items():
                            pct = count / len(target_list_df) * 100
                            st.caption(f"• {val}: {count} ({pct:.1f}%)")
            
            # Target list table
            st.markdown("---")
            st.markdown("### 📋 Target Customer List")
            
            display_cols = [KEY_COL, 'Probability', 'PX']
            if CONTACT_COL in target_list_df.columns:
                display_cols.append(CONTACT_COL)
            display_cols += [f for f in active_demo_features if f in target_list_df.columns]
            
            st.dataframe(
                target_list_df[display_cols].sort_values('Probability', ascending=False).head(100).style.background_gradient(subset=['Probability'], cmap='Greens'),
                use_container_width=True,
                hide_index=True
            )
            
            # Export options
            st.markdown("---")
            st.markdown("### 📥 Export for Campaign")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                # Full CSV
                csv_full = target_list_df[display_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Full List (CSV)",
                    csv_full,
                    f"campaign_targets_{target_product.replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with export_cols[1]:
                # Contact list (phone numbers)
                contact_cols = [KEY_COL, 'Probability']
                if CONTACT_COL in target_list_df.columns:
                    contact_cols.append(CONTACT_COL)
                contact_list = target_list_df[contact_cols].copy()
                csv_contact = contact_list.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Contact List",
                    csv_contact,
                    f"contact_targets_{target_product.replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with export_cols[2]:
                # Summary report
                summary = f"""CAMPAIGN TARGET REPORT
=======================
Product: {target_product}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

SUMMARY:
- Total Targets: {total_targets}
- Hot Leads (>75%): {len(target_list_df[target_list_df['Probability'] > 0.75])}
- Avg Probability: {target_list_df['Probability'].mean()*100:.1f}%
- Excluded Existing Buyers: {exclude_buyers}

TOP DEMOGRAPHICS:
"""
                for feat in active_demo_features[:3]:
                    if feat in target_list_df.columns:
                        top_val = target_list_df[feat].mode()
                        if not top_val.empty:
                            summary += f"- {feat}: {top_val.iloc[0]}\n"
                
                st.download_button(
                    "Download Summary Report",
                    summary,
                    f"campaign_summary_{target_product.replace(' ', '_')}.txt",
                    "text/plain",
                    use_container_width=True
                )
        
        # =================================================================
        # TAB 4: MODEL PERFORMANCE
        # =================================================================
        with tab_model:
            st.markdown("## 📈 Model Performance Metrics")
            
            m1, m2 = st.columns(2)
            m1.metric("AUC-ROC Score", f"{evals['auc']:.4f}")
            m2.metric("Accuracy (Test Set)", f"{evals['report']['accuracy']:.4f}" if 'accuracy' in evals['report'] else "N/A")

            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("**Confusion Matrix**")
                    st.pyplot(evals['cm_plot'], use_container_width=True)
            with c2:
                with st.container(border=True):
                    st.markdown("**ROC Curve**")
                    st.pyplot(evals['roc_plot'], use_container_width=True)
            
            # Interpretation
            auc = evals['auc']
            if auc >= 0.8:
                quality = "Excellent"
                interpretation = "Model sangat baik dalam membedakan pembeli dan non-pembeli"
                color = "#22c55e"
            elif auc >= 0.7:
                quality = "Good"
                interpretation = "Model cukup baik, hasil prediksi dapat diandalkan"
                color = "#3b82f6"
            elif auc >= 0.6:
                quality = "Fair"
                interpretation = "Model masih berguna, tapi ada ruang untuk improvement"
                color = "#f59e0b"
            else:
                quality = "Poor"
                interpretation = "Model kurang akurat, pertimbangkan fitur atau data tambahan"
                color = "#ef4444"
            
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin-top: 20px;">
                <h4 style="margin: 0; color: {color};">Model Quality: {quality}</h4>
                <p style="margin: 10px 0 0 0; color: #475569;">{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)
        
# --- PAGE 6: BUSINESS INSIGHTS ---
elif selected_page == "Business Insights":
    st.markdown('<div class="main-header">💡 Comprehensive Business Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Agregasi dan sintesis terintegrasi atas seluruh keluaran analitik DSS—Association Rules Mining, RFM Segmentation, Product Affinity Analysis, dan Artificial Neural Network—untuk formulasi keputusan strategis berbasis data.</div>', unsafe_allow_html=True)
    
    # Import affinity functions
    from database import (
        get_demographic_options, 
        get_product_affinity_by_demographic,
        get_demographic_distribution
    )
    
    # =============================================================================
    # CHECK ANALYSIS AVAILABILITY
    # =============================================================================
    has_ar = st.session_state.association_rules is not None and not st.session_state.association_rules.empty
    has_rfm = st.session_state.rfm_calculated and st.session_state.rfm_data is not None
    has_ann = 'prediction_results' in st.session_state and st.session_state.prediction_results is not None
    has_data = st.session_state.data_loaded and st.session_state.data is not None
    
    # Status indicators
    st.markdown("### 📊 Status Analisis DSS")
    status_cols = st.columns(4)
    with status_cols[0]:
        if has_ar:
            st.success(f"✅ Association Rules\n{len(st.session_state.association_rules)} pola")
        else:
            st.warning("⚠️ Association Rules\nBelum dijalankan")
    with status_cols[1]:
        if has_rfm:
            st.success(f"✅ RFM Analysis\n{len(st.session_state.rfm_data)} pelanggan")
        else:
            st.warning("⚠️ RFM Analysis\nBelum dijalankan")
    with status_cols[2]:
        if has_ann:
            st.success(f"✅ ANN Prediction\nModel terlatih")
        else:
            st.warning("⚠️ ANN Prediction\nBelum dijalankan")
    with status_cols[3]:
        if has_data:
            st.success(f"✅ Data Ready\n{len(st.session_state.data):,} baris")
        else:
            st.error("❌ Data\nTidak tersedia")
    
    if not has_data:
        st.error("⚠️ Silakan muat data dari Database terlebih dahulu.")
        st.stop()
    
    active_demo_features = get_active_demo_features()
    
    # =============================================================================
    # TABS FOR COMPREHENSIVE INSIGHTS
    # =============================================================================
    tab_summary, tab_ar, tab_rfm, tab_affinity, tab_ann, tab_strategy = st.tabs([
        "📋 Executive Summary", 
        "🛒 Association Rules", 
        "👥 RFM Segmentation",
        "💝 Product Affinity",
        "🤖 ANN Predictions",
        "🎯 Strategic Action Plan"
    ])
    
    # =============================================================================
    # TAB 1: EXECUTIVE SUMMARY
    # =============================================================================
    with tab_summary:
        st.markdown("## 🎯 Executive Summary - Ringkasan Eksekutif")
        
        # --- SMART CONCLUSION BOX ---
        conclusions = []
        strategies = []
        
        # 1. Association Rules Conclusions
        if has_ar:
            rules = st.session_state.association_rules
            top_rule = rules.nlargest(1, 'lift').iloc[0]
            avg_confidence = rules['confidence'].mean() * 100
            high_lift_rules = len(rules[rules['lift'] > 2])
            
            conclusions.append(f"🛒 <b>Pola Belanja:</b> Ditemukan <b>{len(rules)}</b> pola pembelian. "
                             f"Pola terkuat: <b>{top_rule['antecedents_str']}</b> → <b>{top_rule['consequents_str']}</b> "
                             f"(Lift: {top_rule['lift']:.2f}x, Confidence: {top_rule['confidence']*100:.0f}%)")
            
            if high_lift_rules > 5:
                strategies.append("📦 <b>Product Bundling:</b> Buat {0} paket bundling berdasarkan pola dengan lift tinggi".format(min(high_lift_rules, 10)))
            if avg_confidence > 50:
                strategies.append("🏷️ <b>Cross-Selling:</b> Implementasikan rekomendasi produk otomatis di checkout")
        
        # 2. RFM Conclusions
        if has_rfm:
            rfm = st.session_state.rfm_data
            segment_counts = rfm['Segment'].value_counts()
            
            champions = segment_counts.get('🏆 Champions', 0)
            loyal = segment_counts.get('💎 Loyal Customers', 0)
            at_risk = segment_counts.get('🚨 At Risk', 0)
            cant_lose = segment_counts.get("🔥 Can't Lose Them", 0)
            hibernating = segment_counts.get('❄️ Hibernating', 0)
            
            total_customers = len(rfm)
            vip_pct = (champions + loyal) / total_customers * 100
            risk_pct = (at_risk + cant_lose) / total_customers * 100
            
            conclusions.append(f"👥 <b>Nilai Pelanggan:</b> <b>{vip_pct:.1f}%</b> pelanggan adalah VIP (Champions + Loyal). "
                             f"<b>{risk_pct:.1f}%</b> berisiko churn dan butuh perhatian segera.")
            
            if at_risk + cant_lose > 10:
                strategies.append(f"🚨 <b>Win-Back Campaign:</b> Prioritaskan {at_risk + cant_lose} pelanggan berisiko dengan voucher khusus")
            if champions > 0:
                strategies.append(f"👑 <b>VIP Program:</b> Buat program loyalitas eksklusif untuk {champions} Champions")
            if hibernating > total_customers * 0.3:
                strategies.append(f"💤 <b>Reactivation:</b> {hibernating} pelanggan hibernating perlu email reaktivasi")
        
        # 3. ANN Conclusions
        if has_ann:
            pred_df = st.session_state.prediction_results
            unique_pred = pred_df.drop_duplicates(subset=[KEY_COL])
            
            hot_leads = len(unique_pred[unique_pred['Probability'] > 0.75])
            warm_leads = len(unique_pred[(unique_pred['Probability'] > 0.5) & (unique_pred['Probability'] <= 0.75)])
            cold_leads = len(unique_pred[unique_pred['Probability'] <= 0.5])
            
            auc = st.session_state.eval_metrics['auc']
            
            conclusions.append(f"🤖 <b>Prediksi AI:</b> Model ANN (AUC: {auc:.2f}) mengidentifikasi "
                             f"<b>{hot_leads}</b> Hot Leads, <b>{warm_leads}</b> Warm, <b>{cold_leads}</b> Cold.")
            
            if hot_leads > 0:
                strategies.append(f"🔥 <b>Priority Campaign:</b> Target {hot_leads} hot leads dengan penawaran premium")
            if auc < 0.7:
                strategies.append("⚙️ <b>Model Improvement:</b> Pertimbangkan penambahan fitur atau tuning parameter")
        
        # 4. Product Affinity Conclusions (from database)
        try:
            demo_options = get_demographic_options()
            affinity_df, _ = get_product_affinity_by_demographic('INCOME_DESC', 'DEPARTMENT')
            if affinity_df is not None and not affinity_df.empty:
                high_affinity = affinity_df[affinity_df['affinity_index'] > 1.5]
                if not high_affinity.empty:
                    top_affinity = high_affinity.nlargest(1, 'affinity_index').iloc[0]
                    conclusions.append(f"💝 <b>Preferensi Demografis:</b> Segmen <b>{top_affinity['segment']}</b> "
                                     f"memiliki preferensi sangat tinggi untuk <b>{top_affinity['product']}</b> "
                                     f"(Affinity Index: {top_affinity['affinity_index']:.2f})")
                    strategies.append("🎯 <b>Targeted Ads:</b> Sesuaikan iklan berdasarkan preferensi demografis")
        except:
            pass
        
        # Display Smart Conclusion Box
        if conclusions:
            conclusion_html = "<br><br>".join(conclusions)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px;">
                <h3 style="margin-top:0; color: white;">🧠 AI Smart Conclusion</h3>
                <div style="font-size: 15px; line-height: 1.8;">
                    {conclusion_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display Strategy Recommendations
        if strategies:
            strategy_html = "".join([f"<li style='margin-bottom: 10px;'>{s}</li>" for s in strategies])
            st.markdown(f"""
            <div style="background-color: #f0fdf4; border-left: 6px solid #22c55e; 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin-top:0; color: #166534;">🎯 Rekomendasi Strategis</h3>
                <ul style="font-size: 15px; color: #334155;">
                    {strategy_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # --- KEY METRICS DASHBOARD ---
        st.markdown("### 📊 Key Performance Indicators")
        
        kpi_cols = st.columns(5)
        
        with kpi_cols[0]:
            if has_ar:
                st.metric("Pola Ditemukan", f"{len(st.session_state.association_rules)}", 
                         delta=f"Avg Lift: {st.session_state.association_rules['lift'].mean():.2f}x")
            else:
                st.metric("Pola Ditemukan", "N/A", delta="Belum dianalisis")
        
        with kpi_cols[1]:
            if has_rfm:
                champions = len(st.session_state.rfm_data[st.session_state.rfm_data['Segment'] == '🏆 Champions'])
                st.metric("Champions", f"{champions}", delta="Pelanggan terbaik")
            else:
                st.metric("Champions", "N/A", delta="Belum dianalisis")
        
        with kpi_cols[2]:
            if has_rfm:
                at_risk = len(st.session_state.rfm_data[st.session_state.rfm_data['Segment'].isin(['🚨 At Risk', "🔥 Can't Lose Them"])])
                st.metric("At Risk", f"{at_risk}", delta="Perlu perhatian", delta_color="inverse")
            else:
                st.metric("At Risk", "N/A", delta="Belum dianalisis")
        
        with kpi_cols[3]:
            if has_ann:
                unique_pred = st.session_state.prediction_results.drop_duplicates(subset=[KEY_COL])
                hot = len(unique_pred[unique_pred['Probability'] > 0.75])
                st.metric("Hot Leads", f"{hot}", delta="Prioritas tinggi")
            else:
                st.metric("Hot Leads", "N/A", delta="Belum dianalisis")
        
        with kpi_cols[4]:
            if has_ann:
                auc = st.session_state.eval_metrics['auc']
                quality = "Excellent" if auc > 0.8 else "Good" if auc > 0.7 else "Fair"
                st.metric("Model Quality", f"{auc:.2f}", delta=quality)
            else:
                st.metric("Model Quality", "N/A", delta="Belum dianalisis")
        
        # --- OVERALL BUSINESS HEALTH SCORE ---
        st.markdown("### 🏥 Business Health Score")
        
        health_scores = []
        health_details = []
        
        if has_ar:
            ar_score = min(100, len(st.session_state.association_rules) * 2 + 
                          st.session_state.association_rules['lift'].mean() * 10)
            health_scores.append(ar_score)
            health_details.append(f"Association Rules: {ar_score:.0f}/100")
        
        if has_rfm:
            rfm = st.session_state.rfm_data
            seg_counts = rfm['Segment'].value_counts()
            vip = seg_counts.get('🏆 Champions', 0) + seg_counts.get('💎 Loyal Customers', 0)
            risk = seg_counts.get('🚨 At Risk', 0) + seg_counts.get("🔥 Can't Lose Them", 0)
            rfm_score = max(0, min(100, (vip / len(rfm) * 200) - (risk / len(rfm) * 100)))
            health_scores.append(rfm_score)
            health_details.append(f"Customer Health: {rfm_score:.0f}/100")
        
        if has_ann:
            auc = st.session_state.eval_metrics['auc']
            ann_score = auc * 100
            health_scores.append(ann_score)
            health_details.append(f"Prediction Accuracy: {ann_score:.0f}/100")
        
        if health_scores:
            overall_health = sum(health_scores) / len(health_scores)
            
            if overall_health >= 75:
                health_color = "#22c55e"
                health_status = "EXCELLENT - Bisnis dalam kondisi sangat baik"
            elif overall_health >= 50:
                health_color = "#f59e0b"
                health_status = "GOOD - Ada ruang untuk improvement"
            else:
                health_color = "#ef4444"
                health_status = "NEEDS ATTENTION - Perlu tindakan segera"
            
            col_health1, col_health2 = st.columns([1, 2])
            
            with col_health1:
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: white; border-radius: 15px; 
                            border: 3px solid {health_color};">
                    <div style="font-size: 60px; font-weight: bold; color: {health_color};">{overall_health:.0f}</div>
                    <div style="font-size: 16px; color: #666;">Overall Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_health2:
                st.markdown(f"**Status:** {health_status}")
                st.markdown("**Komponen Skor:**")
                for detail in health_details:
                    st.markdown(f"- {detail}")
    
    # =============================================================================
    # TAB 2: ASSOCIATION RULES INSIGHTS
    # =============================================================================
    with tab_ar:
        st.markdown("## 🛒 Association Rules - Pola Pembelian")
        
        if not has_ar:
            st.warning("⚠️ Jalankan Association Rules terlebih dahulu di menu 'Association Rules'")
        else:
            rules = st.session_state.association_rules
            
            # Summary metrics
            ar_m1, ar_m2, ar_m3, ar_m4 = st.columns(4)
            ar_m1.metric("Total Pola", len(rules))
            ar_m2.metric("Avg Support", f"{rules['support'].mean()*100:.2f}%")
            ar_m3.metric("Avg Confidence", f"{rules['confidence'].mean()*100:.1f}%")
            ar_m4.metric("Max Lift", f"{rules['lift'].max():.2f}x")
            
            st.markdown("---")
            
            # Top 10 Rules by different metrics
            col_ar1, col_ar2 = st.columns(2)
            
            with col_ar1:
                st.markdown("### 🔥 Top 10 Pola (by Lift)")
                top_lift = rules.nlargest(10, 'lift')[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
                top_lift_display = top_lift.copy()
                top_lift_display['support'] = (top_lift_display['support'] * 100).round(2).astype(str) + '%'
                top_lift_display['confidence'] = (top_lift_display['confidence'] * 100).round(1).astype(str) + '%'
                top_lift_display['lift'] = top_lift_display['lift'].round(2).astype(str) + 'x'
                top_lift_display.columns = ['Jika Beli', 'Maka Beli', 'Support', 'Confidence', 'Lift']
                st.dataframe(top_lift_display, use_container_width=True, hide_index=True)
            
            with col_ar2:
                st.markdown("### 📈 Top 10 Pola (by Confidence)")
                top_conf = rules.nlargest(10, 'confidence')[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
                top_conf_display = top_conf.copy()
                top_conf_display['support'] = (top_conf_display['support'] * 100).round(2).astype(str) + '%'
                top_conf_display['confidence'] = (top_conf_display['confidence'] * 100).round(1).astype(str) + '%'
                top_conf_display['lift'] = top_conf_display['lift'].round(2).astype(str) + 'x'
                top_conf_display.columns = ['Jika Beli', 'Maka Beli', 'Support', 'Confidence', 'Lift']
                st.dataframe(top_conf_display, use_container_width=True, hide_index=True)
            
            # Strategic Recommendations
            st.markdown("---")
            st.markdown("### 💡 Rekomendasi Strategis dari Association Rules")
            
            # Bundling recommendations
            high_lift = rules[rules['lift'] > 1.5].nlargest(5, 'confidence')
            if not high_lift.empty:
                st.success("**🎁 Rekomendasi Product Bundling:**")
                for i, row in high_lift.iterrows():
                    st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #22c55e;">
                        <b>Bundle #{high_lift.index.get_loc(i)+1}:</b> {row['antecedents_str']} + {row['consequents_str']}<br>
                        <small>Confidence: {row['confidence']*100:.0f}% | Lift: {row['lift']:.2f}x</small><br>
                        <i>💡 Strategi: Buat paket hemat dengan diskon 10-15% untuk kombinasi ini</i>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Cross-sell recommendations
            popular_rules = rules[rules['support'] > rules['support'].median()].nlargest(3, 'lift')
            if not popular_rules.empty:
                st.info("**🛍️ Rekomendasi Cross-Selling di Checkout:**")
                for i, row in popular_rules.iterrows():
                    st.markdown(f"- Saat pelanggan membeli **{row['antecedents_str']}**, tawarkan **{row['consequents_str']}**")
    
    # =============================================================================
    # TAB 3: RFM INSIGHTS
    # =============================================================================
    with tab_rfm:
        st.markdown("## 👥 RFM Analysis - Segmentasi Pelanggan")
        
        if not has_rfm:
            st.warning("⚠️ Jalankan RFM Analysis terlebih dahulu di menu 'RFM Analysis'")
        else:
            rfm = st.session_state.rfm_data
            segment_counts = rfm['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            # Summary
            total_customers = len(rfm)
            avg_recency = rfm['Recency'].mean()
            avg_frequency = rfm['Frequency'].mean()
            avg_monetary = rfm['Monetary'].mean()
            
            rfm_m1, rfm_m2, rfm_m3, rfm_m4 = st.columns(4)
            rfm_m1.metric("Total Pelanggan", f"{total_customers:,}")
            rfm_m2.metric("Avg Recency", f"{avg_recency:.0f} hari")
            rfm_m3.metric("Avg Frequency", f"{avg_frequency:.1f}x")
            rfm_m4.metric("Avg Monetary", f"{avg_monetary:,.0f}")
            
            st.markdown("---")
            
            # Segment distribution chart
            col_rfm1, col_rfm2 = st.columns([1, 1])
            
            with col_rfm1:
                st.markdown("### 📊 Distribusi Segmen")
                fig_rfm = px.pie(segment_counts, values='Count', names='Segment', 
                               title='Customer Segments', hole=0.4)
                fig_rfm.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_rfm, use_container_width=True)
            
            with col_rfm2:
                st.markdown("### 📋 Detail per Segmen")
                # Add percentage
                segment_counts['Percentage'] = (segment_counts['Count'] / total_customers * 100).round(1)
                segment_counts['Percentage'] = segment_counts['Percentage'].astype(str) + '%'
                st.dataframe(segment_counts, use_container_width=True, hide_index=True)
            
            # Strategic recommendations by segment
            st.markdown("---")
            st.markdown("### 💡 Strategi per Segmen Pelanggan")
            
            segment_strategies = {
                '🏆 Champions': {
                    'priority': 'TINGGI',
                    'color': '#ffd700',
                    'insight': 'Baru saja bertransaksi dengan frekuensi tertinggi dan basket besar; sumber referensi dan advokat merek.',
                    'strategy': 'Program VIP eksklusif, early access produk baru, rewards program premium',
                    'action': 'Pertahankan dengan apresiasi dan privilege khusus'
                },
                '💎 Loyal Customers': {
                    'priority': 'TINGGI', 
                    'color': '#0ea5e9',
                    'insight': 'Belanja rutin dengan ritme stabil; sensitif terhadap konsistensi layanan dan apresiasi loyalitas.',
                    'strategy': 'Upselling produk premium, referral program dengan rewards',
                    'action': 'Tingkatkan ke Champions dengan insentif pembelian'
                },
                '🌟 Potential Loyalist': {
                    'priority': 'SEDANG',
                    'color': '#22c55e',
                    'insight': 'Sedang membangun kebiasaan belanja; perlu nurturing untuk mengunci loyalitas.',
                    'strategy': 'Membership program, personalized offers, engagement campaign',
                    'action': 'Bangun hubungan dengan komunikasi konsisten'
                },
                '🆕 New Customers': {
                    'priority': 'SEDANG',
                    'color': '#a855f7',
                    'insight': 'Baru melakukan pembelian perdana; butuh onboarding mulus dan edukasi produk.',
                    'strategy': 'Welcome series email, first-purchase discount, product education',
                    'action': 'Onboarding yang baik untuk meningkatkan retensi'
                },
                '⚠️ Need Attention': {
                    'priority': 'SEDANG',
                    'color': '#f59e0b',
                    'insight': 'Frekuensi mulai turun dan jeda pembelian melebar; sinyal awal churn yang masih bisa dipulihkan.',
                    'strategy': 'Re-engagement email, limited time offers, feedback survey',
                    'action': 'Cari tahu alasan penurunan dan tawarkan solusi'
                },
                '😴 About to Sleep': {
                    'priority': 'WASPADA',
                    'color': '#64748b',
                    'insight': 'Nyaris hibernasi; transaksi terakhir sudah lama dengan frekuensi rendah.',
                    'strategy': 'Wake-up campaign, flash sale notification, "We miss you" email',
                    'action': 'Aktivasi segera sebelum menjadi hibernating'
                },
                '🚨 At Risk': {
                    'priority': 'KRITIS',
                    'color': '#ef4444',
                    'insight': 'Pernah bernilai tinggi namun kini absen panjang; berpotensi churn bila tidak ada intervensi personal.',
                    'strategy': 'Win-back campaign agresif, deep discount, personal outreach',
                    'action': 'Hubungi langsung dan tawarkan insentif besar'
                },
                '❄️ Hibernating': {
                    'priority': 'RENDAH',
                    'color': '#94a3b8',
                    'insight': 'Dormant dalam jangka panjang; hanya layak disentuh dengan kampanye biaya rendah atau otomatis.',
                    'strategy': 'Reactivation email series, "Come back" promo, brand reminder',
                    'action': 'Low-cost reactivation atau fokus ke segmen lain'
                },
                "🔥 Can't Lose Them": {
                    'priority': 'KRITIS',
                    'color': '#dc2626',
                    'insight': 'VIP bernilai tinggi yang sedang menjauh; retention ROI tertinggi berasal dari intervensi proaktif.',
                    'strategy': 'Personal call dari manager, exclusive recovery offer, VIP treatment',
                    'action': 'URGENT - Pelanggan bernilai tinggi yang hampir hilang'
                }
            }
            
            for segment in segment_counts['Segment'].tolist():
                if segment in segment_strategies:
                    info = segment_strategies[segment]
                    count = segment_counts[segment_counts['Segment'] == segment]['Count'].values[0]
                    seg_df = rfm[rfm['Segment'] == segment]
                    share_pct = (count / total_customers * 100) if total_customers else 0
                    avg_r_seg = seg_df['Recency'].mean()
                    avg_f_seg = seg_df['Frequency'].mean()
                    avg_m_seg = seg_df['Monetary'].mean()
                    avg_r_seg = 0 if pd.isna(avg_r_seg) else avg_r_seg
                    avg_f_seg = 0 if pd.isna(avg_f_seg) else avg_f_seg
                    avg_m_seg = 0 if pd.isna(avg_m_seg) else avg_m_seg
                    rfm_profile = f"R≈{avg_r_seg:.0f} hari | F≈{avg_f_seg:.1f}x | M≈{avg_m_seg:,.0f}"
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0; 
                                border-left: 5px solid {info['color']};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <b style="font-size: 16px;">{segment}</b>
                            <span style="background: {info['color']}; color: white; padding: 3px 10px; 
                                        border-radius: 5px; font-size: 12px;">Prioritas: {info['priority']}</span>
                        </div>
                        <div style="color: #475569; margin: 5px 0;">
                            📊 Jumlah: {count} pelanggan ({share_pct:.1f}% dari basis)<br>
                            🧮 Profil RFM: {rfm_profile}
                        </div>
                        <div style="margin: 10px 0; color: #475569;"><b>🧠 Insight:</b> {info['insight']}</div>
                        <div style="margin: 10px 0;"><b>📋 Strategi:</b> {info['strategy']}</div>
                        <div style="color: #059669;"><b>✅ Aksi:</b> {info['action']}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # =============================================================================
    # TAB 4: PRODUCT AFFINITY INSIGHTS
    # =============================================================================
    with tab_affinity:
        st.markdown("## 💝 Product Affinity - Preferensi Demografis")
        
        try:
            demo_options = get_demographic_options()
            
            # Select demographic dimension
            selected_demo = st.selectbox(
                "Pilih Dimensi Demografis untuk Analisis",
                options=list(demo_options.keys()),
                format_func=lambda x: f"{demo_options[x]} ({x})"
            )
            
            # Get affinity data
            affinity_df, aff_err = get_product_affinity_by_demographic(selected_demo, 'DEPARTMENT')
            dist_df, dist_err = get_demographic_distribution(selected_demo)
            
            if affinity_df is not None and not affinity_df.empty:
                # Summary
                segments = affinity_df['segment'].unique()
                products = affinity_df['product'].unique()
                
                aff_m1, aff_m2, aff_m3 = st.columns(3)
                aff_m1.metric("Segmen Demografis", len(segments))
                aff_m2.metric("Kategori Produk", len(products))
                aff_m3.metric("Max Affinity Index", f"{affinity_df['affinity_index'].max():.2f}")
                
                st.markdown("---")
                
                # Top affinity by segment
                st.markdown("### 🎯 Top Product Affinity per Segmen")
                
                for segment in list(segments)[:5]:  # Top 5 segments
                    seg_data = affinity_df[affinity_df['segment'] == segment].nlargest(3, 'affinity_index')
                    
                    if not seg_data.empty:
                        products_str = ", ".join([f"**{row['product']}** ({row['affinity_index']:.2f})" 
                                                 for _, row in seg_data.iterrows()])
                        st.markdown(f"**{segment}:** {products_str}")
                
                st.markdown("---")
                
                # Strategic recommendations
                st.markdown("### 💡 Rekomendasi Targeting Demografis")
                
                high_affinity = affinity_df[affinity_df['affinity_index'] > 1.5].groupby('segment').first().reset_index()
                
                if not high_affinity.empty:
                    for _, row in high_affinity.head(5).iterrows():
                        st.markdown(f"""
                        <div style="background: #fef3c7; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #f59e0b;">
                            <b>🎯 Target: {row['segment']}</b><br>
                            <b>📦 Produk Fokus:</b> {row['product']}<br>
                            <b>📊 Affinity Index:</b> {row['affinity_index']:.2f} (sangat tinggi)<br>
                            <i>💡 Strategi: Alokasikan budget iklan untuk segmen ini dengan fokus produk {row['product']}</i>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Data affinity tidak tersedia. Pastikan database sudah dimuat dengan benar.")
        except Exception as e:
            st.error(f"Error loading affinity data: {e}")
    
    # =============================================================================
    # TAB 5: ANN PREDICTIONS INSIGHTS
    # =============================================================================
    with tab_ann:
        st.markdown("## 🤖 ANN Marketing Intelligence")
        
        if not has_ann:
            st.warning("⚠️ Jalankan ANN Training terlebih dahulu di menu 'ANN Training'")
        else:
            pred_df = st.session_state.prediction_results
            unique_pred = pred_df.drop_duplicates(subset=[KEY_COL])
            evals = st.session_state.eval_metrics
            target_product = st.session_state.get('target_product', 'Unknown')
            
            # Header
            st.markdown(f"**🎯 Produk Target:** `{target_product}`")
            
            # Summary metrics
            ann_m1, ann_m2, ann_m3, ann_m4 = st.columns(4)
            ann_m1.metric("Total Pelanggan", f"{len(unique_pred):,}")
            ann_m2.metric("AUC Score", f"{evals['auc']:.3f}")
            ann_m3.metric("Predicted Buyers", len(unique_pred[unique_pred['Probability'] > 0.6]))
            ann_m4.metric("New Targets", len(unique_pred[(unique_pred['Probability'] > 0.6) & (unique_pred['PX'] == 0)]))
            
            st.markdown("---")
            
            # KEY INSIGHT: Buyer Persona Summary
            st.markdown("### 👤 Target Buyer Persona")
            
            true_positives = unique_pred[unique_pred['Probability'] > 0.6]
            new_targets = true_positives[true_positives['PX'] == 0]  # Never bought before
            persona_summary = []
            
            if not new_targets.empty:
                # Build persona
                for feat in active_demo_features[:5]:
                    if feat in new_targets.columns:
                        top_val = new_targets[feat].mode()
                        if not top_val.empty:
                            pct = (new_targets[feat] == top_val.iloc[0]).sum() / len(new_targets) * 100
                            persona_summary.append({
                                'factor': feat.replace('_', ' ').title(),
                                'value': top_val.iloc[0],
                                'percentage': pct
                            })
                
                if persona_summary:
                    persona_cols = st.columns(len(persona_summary))
                    icons = ['🎂', '💰', '💍', '🏠', '👶']
                    
                    for i, (col, item) in enumerate(zip(persona_cols, persona_summary)):
                        with col:
                            st.markdown(f"""
                            <div style="background: white; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e2e8f0;">
                                <div style="font-size: 24px;">{icons[i] if i < len(icons) else '📊'}</div>
                                <div style="font-size: 12px; color: #64748b;">{item['factor']}</div>
                                <div style="font-weight: bold; color: #1e40af;">{item['value']}</div>
                                <div style="font-size: 11px; color: #22c55e;">{item['percentage']:.0f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Feature Importance Summary
            if 'feature_importance' in st.session_state and st.session_state.feature_importance is not None:
                st.markdown("### 📊 Key Purchase Drivers")
                
                feat_imp = st.session_state.feature_importance
                
                # Group by demographic
                demo_importance = {}
                for _, row in feat_imp.iterrows():
                    for demo in active_demo_features:
                        if row['feature'].startswith(demo):
                            demo_clean = demo.replace('_', ' ').title()
                            if demo_clean not in demo_importance:
                                demo_importance[demo_clean] = 0
                            demo_importance[demo_clean] += max(0, row['importance'])
                            break
                
                if demo_importance:
                    # Sort and get top 5
                    sorted_importance = sorted(demo_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    total_imp = sum(v for _, v in sorted_importance)
                    
                    for demo, imp in sorted_importance:
                        pct = (imp / total_imp * 100) if total_imp > 0 else 0
                        bar_width = int(pct * 3)
                        st.markdown(f"""
                        <div style="margin: 5px 0;">
                            <div style="display: flex; align-items: center;">
                                <div style="width: 150px; font-size: 13px;">{demo}</div>
                                <div style="flex: 1; background: #e2e8f0; border-radius: 5px; height: 20px;">
                                    <div style="background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: {bar_width}%; height: 100%; border-radius: 5px;"></div>
                                </div>
                                <div style="width: 50px; text-align: right; font-size: 12px; color: #64748b;">{pct:.0f}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Actionable Marketing Recommendations
            st.markdown("### 🎯 Actionable Marketing Recommendations")
            
            new_target_count = len(new_targets) if not new_targets.empty else 0
            existing_buyers = len(unique_pred[unique_pred['PX'] == 1])
            
            st.markdown(f"""
            <div style="background: #f0fdf4; padding: 20px; border-radius: 12px; border-left: 5px solid #22c55e;">
                <h4 style="margin: 0 0 15px 0; color: #166534;">✅ Campaign Action Plan untuk "{target_product}"</h4>
                <ol style="margin: 0; padding-left: 20px; color: #334155;">
                    <li><b>Target {new_target_count:,} pelanggan baru</b> yang belum pernah membeli tapi diprediksi akan membeli</li>
                    <li><b>Gunakan profil demografis di atas</b> untuk targeting di Facebook/Google Ads</li>
                    <li><b>Exclude {existing_buyers:,} existing buyers</b> dari campaign (sudah membeli)</li>
                    <li><b>Prioritaskan berdasarkan probability</b> - mulai dari yang tertinggi</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick export
            st.markdown("---")
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                if not new_targets.empty:
                    export_cols = [KEY_COL, 'Probability']
                    if CONTACT_COL in new_targets.columns:
                        export_cols.append(CONTACT_COL)
                    export_cols += [f for f in active_demo_features if f in new_targets.columns]
                    csv_targets = new_targets[export_cols].sort_values('Probability', ascending=False).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        f"Download {new_target_count:,} New Targets",
                        csv_targets,
                        f"new_targets_{target_product.replace(' ', '_')}.csv",
                        "text/csv",
                        use_container_width=True,
                        type="primary"
                    )
            
            with col_export2:
                # Persona summary text
                if persona_summary:
                    persona_text = f"BUYER PERSONA - {target_product}\n"
                    persona_text += "=" * 40 + "\n"
                    for item in persona_summary:
                        persona_text += f"{item['factor']}: {item['value']} ({item['percentage']:.0f}%)\n"
                    persona_text += f"\nTotal New Targets: {new_target_count:,}"
                    
                    st.download_button(
                        "Download Buyer Persona",
                        persona_text,
                        f"buyer_persona_{target_product.replace(' ', '_')}.txt",
                        "text/plain",
                        use_container_width=True
                    )
    
    # =============================================================================
    # TAB 6: STRATEGIC ACTION PLAN
    # =============================================================================
    with tab_strategy:
        st.markdown("## 🎯 Comprehensive Strategic Action Plan")
        st.markdown("Rencana aksi terintegrasi berdasarkan semua hasil analisis DSS")
        
        # Priority matrix
        st.markdown("### 📊 Priority Action Matrix")
        
        actions = []
        
        # Generate actions based on available analyses
        if has_rfm:
            rfm = st.session_state.rfm_data
            seg_counts = rfm['Segment'].value_counts()
            
            at_risk = seg_counts.get('🚨 At Risk', 0) + seg_counts.get("🔥 Can't Lose Them", 0)
            if at_risk > 0:
                actions.append({
                    'priority': 1,
                    'urgency': 'URGENT',
                    'action': f'Win-Back Campaign untuk {at_risk} pelanggan berisiko',
                    'source': 'RFM',
                    'impact': 'HIGH',
                    'effort': 'MEDIUM',
                    'timeline': '1-2 minggu',
                    'details': 'Kirim email personal dengan voucher 20-30% untuk reaktivasi'
                })
            
            champions = seg_counts.get('🏆 Champions', 0)
            if champions > 0:
                actions.append({
                    'priority': 2,
                    'urgency': 'HIGH',
                    'action': f'VIP Program untuk {champions} Champions',
                    'source': 'RFM',
                    'impact': 'HIGH',
                    'effort': 'HIGH',
                    'timeline': '2-4 minggu',
                    'details': 'Buat tier membership dengan benefits eksklusif'
                })
        
        if has_ar:
            rules = st.session_state.association_rules
            high_lift = len(rules[rules['lift'] > 2])
            if high_lift > 0:
                actions.append({
                    'priority': 3,
                    'urgency': 'MEDIUM',
                    'action': f'Buat {min(high_lift, 5)} Product Bundles',
                    'source': 'Association Rules',
                    'impact': 'MEDIUM',
                    'effort': 'LOW',
                    'timeline': '1 minggu',
                    'details': 'Kombinasikan produk dengan lift tinggi, diskon 10-15%'
                })
            
            actions.append({
                'priority': 4,
                'urgency': 'MEDIUM',
                'action': 'Implementasi Cross-Sell di Checkout',
                'source': 'Association Rules',
                'impact': 'MEDIUM',
                'effort': 'MEDIUM',
                'timeline': '2-3 minggu',
                'details': 'Tambahkan rekomendasi "Customers also bought" berdasarkan rules'
            })
        
        if has_ann:
            pred_df = st.session_state.prediction_results.drop_duplicates(subset=[KEY_COL])
            hot = len(pred_df[pred_df['Probability'] > 0.75])
            if hot > 0:
                actions.append({
                    'priority': 2,
                    'urgency': 'HIGH',
                    'action': f'Targeted Campaign ke {hot} Hot Leads',
                    'source': 'ANN Prediction',
                    'impact': 'HIGH',
                    'effort': 'LOW',
                    'timeline': '1 minggu',
                    'details': 'Email blast dengan penawaran produk target'
                })
        
        # Sort by priority
        actions.sort(key=lambda x: x['priority'])
        
        # Display action cards
        for action in actions:
            urgency_color = {'URGENT': '#ef4444', 'HIGH': '#f59e0b', 'MEDIUM': '#3b82f6', 'LOW': '#22c55e'}
            
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 12px; margin: 15px 0; 
                        border-left: 5px solid {urgency_color.get(action['urgency'], '#666')}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 18px; font-weight: bold;">#{action['priority']} {action['action']}</span>
                    <span style="background: {urgency_color.get(action['urgency'], '#666')}; color: white; 
                                padding: 5px 15px; border-radius: 20px; font-size: 12px;">{action['urgency']}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
                    <div><b>📊 Source:</b> {action['source']}</div>
                    <div><b>💥 Impact:</b> {action['impact']}</div>
                    <div><b>⚡ Effort:</b> {action['effort']}</div>
                    <div><b>⏱️ Timeline:</b> {action['timeline']}</div>
                </div>
                <div style="background: #f8fafc; padding: 10px; border-radius: 8px; color: #475569;">
                    💡 {action['details']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Download comprehensive report
        st.markdown("---")
        st.markdown("### 📥 Download Laporan Lengkap")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            if has_rfm:
                csv_rfm = st.session_state.rfm_data.to_csv(index=False).encode('utf-8')
                st.download_button("Download RFM Data", csv_rfm, "rfm_segmentation.csv", "text/csv")
        
        with col_dl2:
            if has_ar:
                csv_ar = st.session_state.association_rules.to_csv(index=False).encode('utf-8')
                st.download_button("Download Association Rules", csv_ar, "association_rules.csv", "text/csv")
        
        with col_dl3:
            if has_ann:
                csv_pred = st.session_state.prediction_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv_pred, "predictions.csv", "text/csv")
        
        # Executive summary for presentation
        st.markdown("---")
        with st.expander("📄 Executive Summary (Copy untuk Presentasi)"):
            summary_text = f"""
# EXECUTIVE SUMMARY - Retail Decision Support System

## 📊 Hasil Analisis:
"""
            if has_ar:
                summary_text += f"""
### Association Rules:
- Total pola ditemukan: {len(st.session_state.association_rules)}
- Rata-rata confidence: {st.session_state.association_rules['confidence'].mean()*100:.1f}%
- Maksimum lift: {st.session_state.association_rules['lift'].max():.2f}x
"""
            if has_rfm:
                rfm = st.session_state.rfm_data
                summary_text += f"""
### RFM Segmentation:
- Total pelanggan: {len(rfm):,}
- Champions: {len(rfm[rfm['Segment'] == '🏆 Champions'])}
- At Risk: {len(rfm[rfm['Segment'].isin(['🚨 At Risk', "🔥 Can't Lose Them"])])}
"""
            if has_ann:
                pred = st.session_state.prediction_results.drop_duplicates(subset=[KEY_COL])
                summary_text += f"""
### ANN Predictions:
- Model AUC: {st.session_state.eval_metrics['auc']:.3f}
- Hot Leads: {len(pred[pred['Probability'] > 0.75])}
"""
            summary_text += """
## 🎯 Rekomendasi Prioritas:
1. Win-back campaign untuk pelanggan at-risk
2. VIP program untuk Champions
3. Product bundling berdasarkan pola belanja
4. Targeted campaign ke hot leads
"""
            st.text_area("Copy text di bawah:", summary_text, height=400)

# --- PAGE 7: DATABASE MANAGEMENT ---
elif selected_page == "Database":
    st.markdown('<div class="main-header">🗄️ Database Management System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Kelola database SQLite untuk analisis data retail yang lebih mendalam.</div>', unsafe_allow_html=True)

    # Check if database exists
    db_exists = db.database_exists()
    
    # --- DATABASE STATUS ---
    col_status, col_action = st.columns([2, 1])
    
    with col_status:
        if db_exists:
            st.success("✅ Database aktif: `datasets/retail.db`")
            table_info = db.get_table_info()
            total_rows = sum(info['row_count'] for info in table_info.values())
            st.caption(f"Total: {len(table_info)} tabel, {total_rows:,} baris")
        else:
            st.warning("⚠️ Database belum dibuat. Klik tombol untuk memuat data dari CSV.")
    
    with col_action:
        if not db_exists:
            if st.button("🚀 Buat Database", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                with st.spinner("Memuat data dari CSV ke SQLite..."):
                    summary = db.load_all_data(update_progress)
                
                progress_bar.empty()
                status_text.empty()
                
                if 'error' not in summary:
                    db.clear_cached_queries()
                    st.success("✅ Database berhasil dibuat!")
                    st.rerun()
                else:
                    st.error(f"Gagal: {summary['error']}")
        else:
            if st.button("🔄 Refresh Database", use_container_width=True):
                db.delete_database()
                db.clear_cached_queries()
                st.rerun()

    if db_exists:
        # --- TABS FOR DIFFERENT VIEWS ---
        tab1, tab2, tab3 = st.tabs(["📊 Table Browser", "🔍 SQL Query", "ℹ️ Schema"])
        
        # === TAB 1: TABLE BROWSER ===
        with tab1:
            table_info = db.get_table_info()
            table_names = list(table_info.keys())
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**📁 Tables**")
                for tbl in table_names:
                    count = table_info[tbl]['row_count']
                    st.caption(f"• `{tbl}` ({count:,} rows)")
            
            with col2:
                selected_table = st.selectbox("Pilih Tabel:", table_names)
                
                if selected_table:
                    # Show sample data
                    df_sample, err = db.get_table_sample(selected_table, limit=100)
                    if err:
                        st.error(f"Error: {err}")
                    else:
                        st.markdown(f"**Preview: `{selected_table}`** (100 baris pertama)")
                        st.dataframe(df_sample, use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv = df_sample.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"Download {selected_table}.csv",
                            csv,
                            f"{selected_table}.csv",
                            "text/csv"
                        )
        
        # === TAB 2: SQL QUERY ===
        with tab2:
            st.markdown("**🔍 Execute Custom SQL Query**")
            st.caption("Tulis query SQL untuk mengeksplorasi data. Gunakan SELECT untuk keamanan.")
            
            # Example queries
            example_queries = {
                "Pilih contoh query...": "",
                "Top 10 Pelanggan (Spending)": """
SELECT 
    c.household_key,
    c.AGE_DESC,
    c.INCOME_DESC,
    COUNT(DISTINCT t.BASKET_ID) as total_transactions,
    ROUND(SUM(t.SALES_VALUE), 2) as total_spend
FROM customers c
JOIN transactions t ON c.household_key = t.household_key
GROUP BY c.household_key
ORDER BY total_spend DESC
LIMIT 10""",
                "Penjualan per Department": """
SELECT 
    p.DEPARTMENT,
    COUNT(DISTINCT t.BASKET_ID) as transactions,
    ROUND(SUM(t.SALES_VALUE), 2) as revenue
FROM products p
JOIN transactions t ON p.PRODUCT_ID = t.PRODUCT_ID
GROUP BY p.DEPARTMENT
ORDER BY revenue DESC""",
                "Campaign Performance": """
SELECT 
    c.CAMPAIGN,
    c.DESCRIPTION,
    COUNT(DISTINCT ct.household_key) as targeted,
    COUNT(DISTINCT cr.household_key) as redeemed,
    ROUND(COUNT(DISTINCT cr.household_key) * 100.0 / 
          NULLIF(COUNT(DISTINCT ct.household_key), 0), 1) as redemption_rate
FROM campaigns c
LEFT JOIN campaign_targets ct ON c.CAMPAIGN = ct.CAMPAIGN
LEFT JOIN coupon_redemptions cr ON c.CAMPAIGN = cr.CAMPAIGN
GROUP BY c.CAMPAIGN
ORDER BY redemption_rate DESC""",
                "Coupon Effectiveness": """
SELECT 
    camp.DESCRIPTION as campaign_type,
    COUNT(DISTINCT cr.household_key) as unique_redeemers,
    COUNT(cr.id) as total_redemptions
FROM coupon_redemptions cr
JOIN campaigns camp ON cr.CAMPAIGN = camp.CAMPAIGN
GROUP BY camp.DESCRIPTION"""
            }
            
            selected_example = st.selectbox("📝 Contoh Query:", list(example_queries.keys()))
            
            default_query = example_queries.get(selected_example, "")
            
            query = st.text_area(
                "SQL Query:",
                value=default_query,
                height=150,
                placeholder="SELECT * FROM customers LIMIT 10"
            )
            
            if st.button("▶️ Execute Query", type="primary"):
                if query.strip():
                    # Basic security check
                    query_upper = query.upper().strip()
                    if not query_upper.startswith("SELECT"):
                        st.error("⚠️ Hanya query SELECT yang diizinkan untuk keamanan.")
                    else:
                        with st.spinner("Executing..."):
                            df_result, err = db.execute_query(query)
                        
                        if err:
                            st.error(f"Error: {err}")
                        else:
                            st.success(f"✅ Query berhasil! {len(df_result)} baris ditemukan.")
                            st.dataframe(df_result, use_container_width=True, hide_index=True)
                            
                            # Download result
                            csv = df_result.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Result", csv, "query_result.csv", "text/csv")
                else:
                    st.warning("Masukkan query SQL terlebih dahulu.")
        
       
        # === TAB 3: SCHEMA ===
        with tab3:
            st.markdown("**ℹ️ Database Schema**")
            
            # --- VISUAL ERD DIAGRAM ---
            st.markdown("### 🗺️ Entity Relationship Diagram")
            
            erd_graph = """
            digraph ERD {
                // Graph settings
                rankdir=LR;
                node [shape=record, fontname="Arial", fontsize=10, style=filled, fillcolor="#e8f4fc", color="#2e7bcf"];
                edge [fontname="Arial", fontsize=9, color="#64748b"];
                graph [bgcolor="transparent", pad=0.5, nodesep=0.8, ranksep=1.2];
                
                // Tables as nodes with key columns
                campaigns [label="{📊 campaigns|CAMPAIGN (PK)\\lDESCRIPTION\\lSTART_DAY\\lEND_DAY\\l}", fillcolor="#dbeafe"];
                
                campaign_targets [label="{🎯 campaign_targets|id (PK)\\lCAMPAIGN (FK)\\lhousehold_key (FK)\\l}", fillcolor="#fef3c7"];
                
                customers [label="{👥 customers|household_key (PK)\\lAGE_DESC\\lMARITAL_STATUS_CODE\\lINCOME_DESC\\lHOMEOWNER_DESC\\lHH_COMP_DESC\\lHOUSEHOLD_SIZE_DESC\\lKID_CATEGORY_DESC\\lphone_number\\l}", fillcolor="#d1fae5"];
                
                coupon_redemptions [label="{✅ coupon_redemptions|id (PK)\\lhousehold_key (FK)\\lDAY\\lCOUPON_UPC\\lCAMPAIGN (FK)\\l}", fillcolor="#fee2e2"];
                
                products [label="{📦 products|PRODUCT_ID (PK)\\lMANUFACTURER\\lDEPARTMENT\\lBRAND\\lCOMMODITY_DESC\\lSUB_COMMODITY_DESC\\lCURR_SIZE_OF_PRODUCT\\l}", fillcolor="#e0e7ff"];
                
                transactions [label="{💳 transactions|id (PK)\\lhousehold_key (FK)\\lBASKET_ID\\lDAY\\lPRODUCT_ID (FK)\\lQUANTITY\\lSALES_VALUE\\lSTORE_ID\\lRETAIL_DISC\\lCOUPON_DISC\\lCOUPON_MATCH_DISC\\l}", fillcolor="#fce7f3"];
                
                // Relationships with labels
                campaigns -> campaign_targets [label="1:N\\nCAMPAIGN", style=bold, color="#2563eb"];
                campaigns -> coupon_redemptions [label="1:N\\nCAMPAIGN", style=bold, color="#2563eb"];
                
                campaign_targets -> customers [label="N:1\\nhousehold_key", style=bold, color="#059669"];
                
                customers -> transactions [label="1:N\\nhousehold_key", style=bold, color="#059669"];
                customers -> coupon_redemptions [label="1:N\\nhousehold_key", style=bold, color="#059669"];
                
                products -> transactions [label="1:N\\nPRODUCT_ID", style=bold, color="#7c3aed"];
            }
            """
            
            st.graphviz_chart(erd_graph, use_container_width=True)
            
            # Legend
            st.markdown("""
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-top: 10px;">
                <strong>📌 Legend:</strong><br>
                <span style="color: #2563eb;">━━ Blue</span>: Campaign relationships &nbsp;&nbsp;
                <span style="color: #059669;">━━ Green</span>: Customer relationships &nbsp;&nbsp;
                <span style="color: #7c3aed;">━━ Purple</span>: Product relationships<br><br>
                <strong>Keys:</strong> PK = Primary Key, FK = Foreign Key
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- TABLE DETAILS ---
            st.markdown("### 📋 Table Details")
            
            table_info = db.get_table_info()
            
            # Create columns for compact view
            col1, col2 = st.columns(2)
            tables_list = list(table_info.items())
            
            for i, (tbl_name, tbl_data) in enumerate(tables_list):
                with (col1 if i % 2 == 0 else col2):
                    with st.expander(f"📁 {tbl_name} ({tbl_data['row_count']:,} rows)"):
                        col_df = pd.DataFrame(tbl_data['columns'], columns=['Column', 'Type'])
                        st.dataframe(col_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # --- RELATIONSHIP SUMMARY ---
            st.markdown("### 🔗 Relationship Summary")
            
            relationships = [
                {"From Table": "campaigns", "To Table": "campaign_targets", "Join Key": "CAMPAIGN", "Type": "1:N"},
                {"From Table": "campaigns", "To Table": "coupon_redemptions", "Join Key": "CAMPAIGN", "Type": "1:N"},
                {"From Table": "customers", "To Table": "campaign_targets", "Join Key": "household_key", "Type": "1:N"},
                {"From Table": "customers", "To Table": "transactions", "Join Key": "household_key", "Type": "1:N"},
                {"From Table": "customers", "To Table": "coupon_redemptions", "Join Key": "household_key", "Type": "1:N"},
                {"From Table": "products", "To Table": "transactions", "Join Key": "PRODUCT_ID", "Type": "1:N"},
            ]
            
            rel_df = pd.DataFrame(relationships)
            st.dataframe(rel_df, use_container_width=True, hide_index=True)
>>>>>>> repo-bim-dss/main
