import streamlit as st
import pandas as pd
import time
from streamlit_option_menu import option_menu
import numpy as np

# --- ERROR HANDLING UNTUK MODUL CUSTOM ---
try:
    import preprocessing as pp
    import model_utils as mu
except ImportError as e:
    st.error(f"❌ Modul custom tidak ditemukan: {e}. Pastikan file 'preprocessing.py' dan 'model_utils.py' ada di folder yang sama.")
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
if 'key_col' not in st.session_state: st.session_state.key_col = None
if 'product_list_col' not in st.session_state: st.session_state.product_list_col = None
if 'demo_features' not in st.session_state: st.session_state.demo_features = None

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
        options=["Upload Data", "Association Rules", "ANN Training", "Prediction Results", "Business Insights"],
        icons=["cloud-upload", "diagram-3", "cpu", "graph-up-arrow", "lightbulb"],
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
    if st.session_state.data_loaded:
        st.success(f"✅ Data Ready\n\nRows: {st.session_state.data.shape[0]}")
    else:
        st.info("ℹ️ Menunggu Data")
    st.markdown('<div class="sidebar-footer">© 2025 Project Dashboard</div>', unsafe_allow_html=True)

# =============================================================================
# 4. HALAMAN UTAMA (LOGIC)
# =============================================================================

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
    else:
        c1, c2 = st.columns([1, 1], gap="large")
        
        with c1:
            st.info("🎯 **Target Prediksi**")
            with st.container(border=True):
                ant_opts = [""] + (st.session_state.antecedents if st.session_state.antecedents else [])
                sel_ant = st.selectbox("Pilih dari pola populer (Opsional):", ant_opts)
                
                target_str = st.text_input("Atau ketik Produk Target (koma separator):", value=sel_ant if sel_ant else "")
                target_list = set(p.strip().upper() for p in target_str.split(',') if p)

        with c2:
            st.success("⚙️ **Parameter Model**")
            with st.container(border=True):
                resample = st.selectbox("Penanganan Data Tidak Seimbang:", 
                                      ['oversampling', 'undersampling'], 
                                      format_func=lambda x: "SMOTE (Oversampling) - Recommended" if x == 'oversampling' else "Random Undersampling")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Mulai Training Model", type="primary", use_container_width=True):
            if not target_list:
                st.error("Tentukan produk target terlebih dahulu!")
            else:
                with st.spinner("🤖 Sedang melatih model (Pre-processing > Encoding > Training)..."):
                    try:
                        df = st.session_state.data
                        p_col = st.session_state.product_list_col
                        d_feats = st.session_state.demo_features
                        
                        data_ann = pp.convert_product_list(df.copy(), p_col)
                        data_target = pp.create_target_variable(data_ann, p_col, target_list)
                        data_enc = pp.encode_features(data_target, d_feats)

                        y_full = data_enc['PX']
                        orig_cols = set(data_target.columns)
                        final_cols = set(data_enc.columns)
                        X_full = data_enc[list(final_cols - orig_cols)].copy() 

                        st.session_state.X_full = X_full
                        st.session_state.full_keys = data_enc[[st.session_state.key_col, p_col, 'PX']]

                        X_train, y_train, X_test, y_test = mu.split_and_resample(X_full, y_full, method=resample)
                        
                        with st.expander("Lihat Distribusi Data Training"):
                            st.write("Target Distribution (Train):", y_train.value_counts())
                        
                        model = mu.train_ann_model(X_train, y_train)
                        st.session_state.model = model
                        st.session_state.eval_metrics = mu.generate_evaluation_metrics(model, X_test, y_test)
                        
                        probs, preds = mu.get_predictions(model, X_full)
                        res_df = st.session_state.full_keys.copy()
                        res_df['Probability'] = probs
                        res_df['Prediction'] = preds
                        st.session_state.prediction_results = res_df
                        
                        st.success("✅ Training Selesai! Lihat hasil detail di menu 'Prediction Results'.")
                        
                    except Exception as e:
                        st.error(f"Gagal training: {e}")

# --- PAGE 4: RESULTS ---
elif selected_page == "Prediction Results":
    st.markdown('<div class="main-header">📈 Model Evaluation & Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analisis performa model dan hasil prediksi pada seluruh pelanggan.</div>', unsafe_allow_html=True)

    if not st.session_state.model:
        st.info("⚠️ Model belum dilatih. Silakan ke menu ANN Training.")
    else:
        evals = st.session_state.eval_metrics
        
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