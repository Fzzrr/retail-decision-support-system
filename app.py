# app.py
import streamlit as st
import pandas as pd
import time 

# Impor fungsi kustom kita
import preprocessing as pp
import model_utils as mu

# --- Konfigurasi Halaman & Session State ---
st.set_page_config(
    page_title="Dashboard Market Basket Analysis & ANN",
    layout="wide"
)

# Fungsi untuk Load CSS dari folder assets
def loadd_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state untuk menyimpan data dan status
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'association_rules' not in st.session_state:
    st.session_state.association_rules = None
if 'antecedents' not in st.session_state:
    st.session_state.antecedents = None

# =============================================================================
# --- SIDEBAR (Input Pengguna) ---
# =============================================================================

st.sidebar.title("Market Basket Analysis & ANN Training")

# Inisialisasi halaman aktif di session_state
if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"   # default halaman pertama

# Tombol-tombol navigasi
if st.sidebar.button("Upload & Konfigurasi Data", use_container_width=True):
    st.session_state.current_page = "upload"

if st.sidebar.button("Association Rules (ARM)", use_container_width=True):
    st.session_state.current_page = "arm"

if st.sidebar.button("Train & Evaluasi ANN", use_container_width=True):
    st.session_state.current_page = "ann"

if st.sidebar.button("Lihat Hasil Prediksi", use_container_width=True):
    st.session_state.current_page = "results"
    
page = st.session_state.current_page

# Main Area berdasarkan halaman aktif

if page == "upload":
    st.header("Upload & Konfigurasi Data")

    uploaded_file = st.file_uploader("Upload file CSV (Gabungan)", type="csv")

    # Load + preprocess data sekali saja
    if uploaded_file and not st.session_state.data_loaded:
        with st.spinner("Memuat dan membersihkan data..."):
            data = pp.load_and_preprocess_data(uploaded_file)
            st.session_state.data = data
            st.session_state.data_loaded = True
        st.success("Data berhasil dimuat!")

    if not st.session_state.data_loaded:
        st.info("Silakan upload file CSV untuk memulai.")
    else:
        df = st.session_state.data
        st.subheader("Preview Data")
        st.dataframe(df.head(100))

        all_columns = df.columns.tolist()

        # Helper untuk default kolom
        def find_default(cols, names):
            for name in names:
                if name in cols:
                    return cols.index(name)
            return 0  # default ke kolom pertama

        # Pilih kolom kunci & list produk
        st.subheader("Konfigurasi Kolom Utama")
        key_col = st.selectbox(
            "Kolom Kunci (ID Customer/Household)", 
            all_columns, 
            index=find_default(all_columns, ['customer_id', 'household_key', 'user_id'])
        )
        product_list_col = st.selectbox(
            "Kolom List Produk (harus format string list)", 
            all_columns, 
            index=find_default(all_columns, ['product_list', 'products', 'items'])
        )

        # Tentukan fitur demografi
        numeric_ids = ['basket_id', 'BASKET_ID', 'transaction_id'] 
        exclude_cols = [key_col, product_list_col, 'PX'] + numeric_ids

        available_features = [
            col for col in all_columns 
            if col not in exclude_cols
        ]

        st.subheader("Pilih Fitur Demografi/Profiling untuk Training")
        demo_features = st.multiselect(
            "Fitur yang akan digunakan untuk training:", 
            available_features, 
            default=available_features
        )

        # Simpan ke session_state untuk dipakai di halaman lain
        st.session_state.key_col = key_col
        st.session_state.product_list_col = product_list_col
        st.session_state.demo_features = demo_features

        st.success("Konfigurasi kolom tersimpan. Lanjut ke halaman 'Association Rules (ARM)'.")

elif page == "arm":
    st.header("Association Rules (ARM)")
    if not st.session_state.data_loaded:
        st.warning("Silahkan upload data terlebih dahulu di halaman Upload & Konfigurasi Data")
    elif st.session_state.product_list_col is None:
        st.warning("Kolom list produk belum dikonfigurasi. Kembali ke halaman 'Upload & Konfigurasi Data'.")
    else:
        df = st.session_state.data
        product_list_col = st.session_state.product_list_col

        st.subheader("Pengaturan ARM (FPGrowth)")
        min_support_val = st.slider(
            "Minimum Support FPGrowth", 
            min_value=0.001, 
            max_value=0.1, 
            value=0.01, 
            step=0.001,
            format="%.3f",
            help="Support yang lebih rendah = lebih banyak rules, tapi bisa lebih lama."
        )

        if st.button("Jalankan Association Rules (ARM)", type="primary"):
            with st.spinner(f"Mengonversi kolom '{product_list_col}'..."):
                data_processed_arm = pp.convert_product_list(df.copy(), product_list_col)
            
            with st.spinner(f"Menjalankan FPGrowth (min_support={min_support_val})..."):
                rules, antecedents = pp.run_association_rules(
                    data_processed_arm, 
                    product_list_col, 
                    min_support=min_support_val
                )
                st.session_state.association_rules = rules
                st.session_state.antecedents = antecedents
                st.success("Association Rules selesai!")

        # Tampilkan hasil ARM jika sudah ada
        if st.session_state.association_rules is not None:
            rules = st.session_state.association_rules
            st.subheader("Hasil Langkah 1: Association Rules (ARM)")
            if rules.empty:
                st.warning("Tidak ada rules yang ditemukan dengan pengaturan support saat ini.")
            else:
                st.info(
                    f"Ditemukan {len(rules)} aturan menarik. "
                    "Gunakan tabel ini untuk menentukan target ANN Anda (halaman 'Train & Evaluasi ANN')."
                )
                display_cols = [
                    'antecedents_str', 
                    'consequents_str', 
                    'support', 
                    'confidence', 
                    'lift'
                ]
                valid_cols = [col for col in display_cols if col in rules.columns]
                st.dataframe(rules[valid_cols])

elif page == "ann":
    
    st.header("ANN & Train Model")
    
    if not st.session_state.data_loaded:
        st.warning("Silahkan upload data terlebih dahulu di halaman Upload & Konfigurasi Data")
    elif st.session_state.key_col is None or st.session_state.product_list_col is None:
        st.warning("Konfigurasi kolom belum lengkap. Kembali ke halaman 'Upload & Konfigurasi Data'.")
    elif st.session_state.demo_features is None or len(st.session_state.demo_features) == 0:
        st.warning("Fitur demografi belum dipilih. Kembali ke halaman 'Upload & Konfigurasi Data'.")
    else:
        df = st.session_state.data
        key_col = st.session_state.key_col
        product_list_col = st.session_state.product_list_col
        demo_features = st.session_state.demo_features

        st.info("Pilih target berdasarkan hasil ARM (halaman 2), lalu masukkan di bawah.")

        # Tampilkan dropdown antecedent populer (jika ada)
        selected_antecedent = ""
        if st.session_state.antecedents:
            selected_antecedent = st.selectbox(
                "Pilih antecedent populer (opsional):",
                options=[""] + st.session_state.antecedents,
            )

        if selected_antecedent:
            default_target_str = selected_antecedent
        else:
            default_target_str = ""

        target_products_str = st.text_input(
            "Masukkan Produk Target (pisahkan koma)", 
            default_target_str
        )
        target_list = set(p.strip().upper() for p in target_products_str.split(',') if p)

        # Pilihan metode resampling
        resampling_method = st.selectbox(
            "Metode Resampling Data Latih:",
            options=['undersampling', 'oversampling'],
            index=1,  # default SMOTE
            format_func=lambda x: "SMOTE (Oversampling)" if x == 'oversampling' else "Random (Undersampling)",
            help="Oversampling (SMOTE) disarankan untuk data imbalance."
        )

        if st.button("TRAIN ANN MODEL", type="primary"):
            if not target_list:
                st.error("Harap tentukan Produk Target terlebih dahulu.")
            else:
                st.session_state.model = None

                # 1. Konversi product list
                with st.spinner(f"Mengonversi kolom '{product_list_col}'..."):
                    data_processed_ann = pp.convert_product_list(df.copy(), product_list_col)

                # 2. Buat variabel target PX
                with st.spinner(f"Membuat variabel target 'PX' untuk {target_products_str}..."):
                    data_with_target = pp.create_target_variable(
                        data_processed_ann, product_list_col, target_list
                    )

                # 3. Encode fitur demografi
                with st.spinner("Melakukan encoding fitur demografi..."):
                    original_cols = set(data_with_target.columns)
                    data_encoded = pp.encode_features(data_with_target, demo_features)

                    y_data = data_encoded['PX']
                    encoded_cols = set(data_encoded.columns)
                    new_encoded_features = list(encoded_cols - original_cols)

                    if not new_encoded_features:
                        st.error("Tidak ada fitur demografi yang di-encode. Model tidak bisa dilatih.")
                        st.stop()

                    X_data = data_encoded[new_encoded_features].copy()

                    st.session_state.X_full = X_data
                    st.session_state.y_full = y_data
                    st.session_state.full_data_with_keys = data_encoded[[key_col, product_list_col, 'PX']]

                # 4. Train-test split & resampling
                with st.spinner("Melakukan Train-Test Split dan Resampling..."):
                    X_train, y_train, X_test, y_test = mu.split_and_resample(
                        st.session_state.X_full, 
                        st.session_state.y_full,
                        method=resampling_method
                    )
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                    st.subheader("Info Debugging (Setelah Resampling)")
                    st.write("Distribusi Y_Train (Data Latih - Setelah Resampling):")
                    st.dataframe(pd.DataFrame(y_train.value_counts(), columns=["count"]))
                    st.write("Distribusi Y_Test (Data Uji - Asli/Imbalanced):")
                    st.dataframe(pd.DataFrame(y_test.value_counts(), columns=["count"]))
                    st.write("Contoh X_Train:")
                    st.dataframe(X_train.head())

                # 5. Latih model ANN
                with st.spinner("MELATIH MODEL ANN... (Bisa memakan waktu)"):
                    start_time = time.time()
                    model = mu.train_ann_model(X_train, y_train)
                    end_time = time.time()

                    st.session_state.model = model
                    st.success(f"Pelatihan ANN selesai dalam {end_time - start_time:.2f} detik.")

                # 6. Evaluasi model pada test set
                with st.spinner("Menghasilkan metrik evaluasi..."):
                    st.session_state.eval_metrics = mu.generate_evaluation_metrics(
                        model, 
                        st.session_state.X_test, 
                        st.session_state.y_test
                    )

                # 7. Prediksi ke seluruh data
                with st.spinner("Membuat prediksi untuk semua data..."):
                    full_probs, full_preds = mu.get_predictions(
                        model, 
                        st.session_state.X_full
                    )

                    results_df = st.session_state.full_data_with_keys.copy()
                    results_df['Probabilitas_Beli_PX'] = full_probs
                    results_df['Prediksi_Beli_PX'] = full_preds
                    st.session_state.prediction_results = results_df

                st.success("Pelatihan ANN dan Prediksi Selesai! Lihat halaman 'Lihat Hasil Prediksi'.")


elif page == "results":
    st.header("Hasil Prediksi ANN")
    if 'model' not in st.session_state or st.session_state.model is None:
        st.info("Model belum dilatih, Silakan latih terlkebih dahulu")
    else:
        eval_data = st.session_state.eval_metrics

        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC-ROC Score", f"{eval_data['auc']:.4f}")
            st.subheader("Classification Report (Test Set)")
            st.dataframe(pd.DataFrame(eval_data['report']).transpose())

        with col2:
            st.subheader("Confusion Matrix (Test Set)")
            st.pyplot(eval_data['cm_plot'])

        st.subheader("ROC Curve (Test Set)")
        st.pyplot(eval_data['roc_plot'])

        st.subheader("Prediksi ANN pada Keseluruhan Data")
        st.dataframe(st.session_state.prediction_results)

        @st.cache_data
        def convert_df_to_csv(_df):
            return _df.to_csv(index=False).encode('utf-8')

        if st.session_state.prediction_results is not None:
            csv_data = convert_df_to_csv(st.session_state.prediction_results)
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv_data,
                file_name="prediksi_live_model.csv",
                mime="text/csv",
            )
