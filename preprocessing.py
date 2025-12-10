# preprocessing.py
import pandas as pd
import ast
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Memuat, membersihkan, dan MENGUBAH NAMA kolom data agar lebih bisnis-friendly.
    """
    df = pd.read_csv(uploaded_file)
    
    # 1. Bersihkan Data (Hapus Baris yang memiliki nilai kosong/NaN)
    df.dropna(inplace=True)
    
    # ==========================================================
    # --- 2. RENAMING KOLOM (Agar Dashboard Lebih Cantik) ---
    # ==========================================================
    # Mapping nama kolom asli (CSV) -> Nama baru (Dashboard)
    column_mapping = {
        # Kolom Identitas
        "BASKET_ID": "ID Transaksi",
        "household_key": "ID Pelanggan",
        "DAY": "Hari Ke",
        
        # Kolom Produk
        "product_list": "Keranjang Belanja",
        
        # Kolom Demografi (Penting untuk Business Insight)
        "AGE_DESC": "Rentang Usia",
        "MARITAL_STATUS_CODE": "Status Pernikahan",
        "INCOME_DESC": "Estimasi Pendapatan",
        "HOMEOWNER_DESC": "Status Rumah",
        "HH_COMP_DESC": "Komposisi Keluarga",
        "HOUSEHOLD_SIZE_DESC": "Ukuran Keluarga",
        "KID_CATEGORY_DESC": "Kategori Anak"
    }
    
    # Terapkan perubahan nama (errors='ignore' agar tidak crash jika kolom tidak ada)
    df.rename(columns=column_mapping, inplace=True)
    
    return df

def convert_product_list(df, product_list_col):
    """
    Mengonversi kolom produk (string) menjadi list Python asli.
    Contoh: "['Roti', 'Susu']" -> ['Roti', 'Susu']
    """
    try:
        # ast.literal_eval aman mengevaluasi string menjadi list
        df[product_list_col] = df[product_list_col].apply(ast.literal_eval)
        
        # Verifikasi bahwa baris pertama berhasil jadi list
        if not isinstance(df[product_list_col].iloc[0], list):
            raise ValueError("Hasil konversi bukan list.")
            
    except (ValueError, SyntaxError, TypeError) as e:
        st.error(f"""
        ❌ Gagal mengonversi kolom '{product_list_col}'.
        Pastikan format di CSV adalah string list Python, contoh: "['A', 'B']".
        Error: {e}
        """)
        st.stop()
    
    return df

@st.cache_data
def run_association_rules(_df, product_list_col, min_support=0.01):
    """
    Menjalankan algoritma FP-Growth untuk mencari pola pembelian.
    """
    # 1. Ambil data transaksi yang valid (List tidak kosong)
    # Menggunakan mask boolean agar lebih aman dan cepat
    mask = _df[product_list_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
    transactions = _df[product_list_col][mask].tolist()
    
    if not transactions:
        st.warning("⚠️ Tidak ada data transaksi yang valid untuk diproses.")
        return pd.DataFrame(), []

    # 2. One-Hot Encoding (Format Wajib FP-Growth)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 3. Jalankan FP-Growth
    # Low_memory=True opsional, tapi use_colnames=True wajib
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        st.warning(f"⚠️ Tidak ditemukan pola dengan Min Support {min_support}. Coba turunkan nilainya.")
        return pd.DataFrame(), []

    # 4. Generate Rules (Berdasarkan Lift)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if rules.empty:
        st.warning("⚠️ Tidak ditemukan aturan asosiasi yang kuat (Lift > 1.0).")
        return pd.DataFrame(), []
    
    # 5. Filter Rules "Menarik" (Opsional, agar user tidak pusing lihat ribuan rules)
    # Kita ambil yang confidence > 30% dan Lift > 1.1
    interesting_rules = rules[
        (rules['confidence'] > 0.3) & 
        (rules['lift'] > 1.1)
    ].sort_values(by='lift', ascending=False)
    
    # Jika hasil filter kosong, kembalikan semua rules saja
    if interesting_rules.empty:
        interesting_rules = rules.sort_values(by='lift', ascending=False)

    # 6. Formatting Tampilan (Set -> String)
    # Mengubah frozenset({'Roti'}) menjadi string "Roti" agar rapi di tabel
    interesting_rules['antecedents_str'] = interesting_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    interesting_rules['consequents_str'] = interesting_rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Ambil daftar unik produk pemicu untuk dropdown
    unique_antecedents = interesting_rules['antecedents_str'].unique().tolist()
    
    return interesting_rules, unique_antecedents

def create_target_variable(df, product_list_col, target_product_list):
    """
    Membuat variabel target 'PX' (1 jika beli target, 0 jika tidak).
    Menggunakan himpunan (set) agar pencarian SANGAT CEPAT.
    """
    target_set = set(target_product_list)
    
    def check_target(cart_items):
        # Cek apakah target_set ada di dalam keranjang belanja ini
        return 1 if target_set.issubset(set(cart_items)) else 0
    
    df['PX'] = df[product_list_col].apply(check_target)
    return df

def encode_features(df, demographic_features):
    """
    Mengubah data kategori (teks) menjadi angka (One-Hot Encoding).
    Contoh: "Menikah" -> 1, "Lajang" -> 0
    """
    if not demographic_features:
        return df
    
    # drop_first=True untuk mengurangi redundansi kolom (Dummy Variable Trap)
    df_encoded = pd.get_dummies(df, columns=demographic_features, drop_first=True)
    
    return df_encoded