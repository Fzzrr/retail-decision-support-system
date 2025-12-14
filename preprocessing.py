# preprocessing.py
import pandas as pd
import ast
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Memuat dan membersihkan data. Tidak ada renaming - menggunakan kolom asli.
    """
    df = pd.read_csv(uploaded_file)
    
    # Bersihkan Data (Hapus Baris yang memiliki nilai kosong/NaN)
    df.dropna(inplace=True)
    
    return df

def convert_product_list(df, product_list_col):
    """
    Mengonversi kolom produk (string) menjadi list Python asli.
    Mendukung dua format:
    1. Python list string: "['Roti', 'Susu']" -> ['Roti', 'Susu']
    2. Comma-separated string (dari GROUP_CONCAT): "Roti,Susu,Teh" -> ['Roti', 'Susu', 'Teh']
    """
    def parse_product_string(val):
        # Handle None or NaN
        if pd.isna(val) or val is None:
            return []
        
        val = str(val).strip()
        
        # Jika kosong
        if not val:
            return []
        
        # Coba format 1: Python list string "['A', 'B']"
        if val.startswith('[') and val.endswith(']'):
            try:
                result = ast.literal_eval(val)
                if isinstance(result, list):
                    return [str(item).strip().upper() for item in result if item]
            except:
                pass
        
        # Format 2: Comma-separated string "A,B,C" (dari GROUP_CONCAT)
        items = [item.strip().upper() for item in val.split(',') if item.strip()]
        return items
    
    try:
        st.caption(f"🔄 Mengonversi {len(df):,} baris data produk...")
        df[product_list_col] = df[product_list_col].apply(parse_product_string)
        
        # Verifikasi bahwa hasilnya adalah list
        sample = df[product_list_col].iloc[0] if len(df) > 0 else []
        if not isinstance(sample, list):
            raise ValueError("Hasil konversi bukan list.")
        
        # Info ke user
        non_empty = df[product_list_col].apply(len).gt(0).sum()
        st.caption(f"✅ Berhasil konversi {non_empty:,} baris dengan produk valid.")
            
    except Exception as e:
        st.error(f"""
        ❌ Gagal mengonversi kolom '{product_list_col}'.
        Error: {e}
        """)
        st.stop()
    
    return df

def run_association_rules(_df, product_list_col, min_support=0.01, min_confidence=0.3, min_lift=1.1):
    """
    Menjalankan algoritma FP-Growth untuk mencari pola pembelian.
    
    Parameters:
    -----------
    _df : DataFrame - Data dengan kolom product_list
    product_list_col : str - Nama kolom yang berisi list produk
    min_support : float - Minimum support (default 0.01 = 1%)
    min_confidence : float - Minimum confidence (default 0.3 = 30%)
    min_lift : float - Minimum lift (default 1.1)
    """
    # Progress feedback
    progress_bar = st.progress(0, text="🔄 Memulai proses FP-Growth...")
    
    # 1. Ambil data transaksi yang valid (List tidak kosong)
    progress_bar.progress(5, text="📋 Memvalidasi data transaksi...")
    mask = _df[product_list_col].apply(lambda x: isinstance(x, list) and len(x) > 0)
    transactions = _df[product_list_col][mask].tolist()
    
    if not transactions:
        progress_bar.empty()
        st.warning("⚠️ Tidak ada data transaksi yang valid untuk diproses.")
        return pd.DataFrame(), []
    
    # Diagnostik: hitung jumlah item unik
    all_items = set(item for tx in transactions for item in tx)
    n_items = len(all_items)
    n_transactions = len(transactions)
    
    st.caption(f"📊 Statistik: {n_transactions:,} transaksi, {n_items:,} item unik")
    
    # Peringatan jika data terlalu besar
    if n_items > 500:
        st.warning(f"⚠️ Jumlah item unik sangat banyak ({n_items:,}). Proses mungkin lambat. Pertimbangkan menggunakan level produk yang lebih tinggi (misal: DEPARTMENT).")
    
    if min_support < 0.005 and n_transactions > 10000:
        st.warning(f"⚠️ Min Support sangat rendah ({min_support}) dengan data besar. Ini dapat menyebabkan proses sangat lama.")

    # 2. One-Hot Encoding (Format Wajib FP-Growth)
    progress_bar.progress(15, text="🔢 Melakukan One-Hot Encoding...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Memory check
    mem_mb = df_encoded.memory_usage(deep=True).sum() / 1024 / 1024
    st.caption(f"💾 Ukuran matrix encoded: {df_encoded.shape[0]:,} × {df_encoded.shape[1]:,} ({mem_mb:.1f} MB)")
    
    # 3. Jalankan FP-Growth
    progress_bar.progress(30, text="⛏️ Menjalankan algoritma FP-Growth (langkah terlama)...")
    try:
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ FP-Growth gagal: {e}")
        return pd.DataFrame(), []
    
    if frequent_itemsets.empty:
        progress_bar.empty()
        st.warning(f"⚠️ Tidak ditemukan pola dengan Min Support {min_support}. Coba turunkan nilainya.")
        return pd.DataFrame(), []
    
    st.caption(f"🔍 Ditemukan {len(frequent_itemsets):,} frequent itemsets")

    # 4. Generate Rules (Berdasarkan Lift)
    progress_bar.progress(70, text="📐 Menghasilkan association rules...")
    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Gagal membuat rules: {e}")
        return pd.DataFrame(), []
    
    if rules.empty:
        progress_bar.empty()
        st.warning(f"⚠️ Tidak ditemukan aturan asosiasi dengan Lift > {min_lift}.")
        return pd.DataFrame(), []
    
    # 5. Filter Rules berdasarkan confidence
    progress_bar.progress(85, text="🎯 Memfilter rules berdasarkan confidence...")
    interesting_rules = rules[
        rules['confidence'] >= min_confidence
    ].sort_values(by='lift', ascending=False)
    
    # Jika hasil filter kosong, kembalikan semua rules saja
    if interesting_rules.empty:
        st.warning(f"⚠️ Tidak ada rules dengan confidence >= {min_confidence}. Menampilkan semua rules.")
        interesting_rules = rules.sort_values(by='lift', ascending=False)

    # 6. Formatting Tampilan (Set -> String)
    progress_bar.progress(95, text="✨ Memformat hasil akhir...")
    interesting_rules['antecedents_str'] = interesting_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    interesting_rules['consequents_str'] = interesting_rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Ambil daftar unik produk pemicu untuk dropdown
    unique_antecedents = interesting_rules['antecedents_str'].unique().tolist()
    
    # Selesai
    progress_bar.progress(100, text="✅ Selesai!")
    progress_bar.empty()
    
    return interesting_rules, unique_antecedents

def create_target_variable(df, product_list_col, target_product_list):
    """
    Membuat variabel target 'PX' dengan PENGAMAN TIPE DATA.
    """
    # Menggunakan set() untuk pencarian yang jauh lebih cepat
    target_set = set(target_product_list)
    
    def check_target(cart_items):
        # --- PENGAMAN (SAFETY CHECK) ---
        # Jika data BUKAN list (misal: angka, atau kosong), anggap tidak beli (0)
        if not isinstance(cart_items, (list, set, tuple)):
            return 0
            
        # Cek apakah target_set ada di dalam keranjang belanja ini
        return 1 if target_set.issubset(set(cart_items)) else 0
    
    # Terapkan fungsi dengan aman
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

# =============================================================================
# RFM ANALYSIS FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner="📊 Menghitung RFM...")
def calculate_rfm(_df, key_col, day_col, product_list_col):
    """
    Menghitung RFM (Recency, Frequency, Monetary) per pelanggan.
    
    Parameters:
    -----------
    _df : DataFrame - Data transaksi
    key_col : str - Nama kolom ID pelanggan
    day_col : str - Nama kolom hari/tanggal transaksi (atau ID Transaksi sebagai proxy)
    product_list_col : str - Nama kolom daftar produk
    
    Returns:
    --------
    DataFrame dengan kolom: customer_id, Recency, Frequency, Monetary, R_Score, F_Score, M_Score, RFM_Score, Segment
    """
    df = _df.copy()
    
    # Pastikan product_list sudah dalam format list
    if df[product_list_col].dtype == 'object':
        try:
            df[product_list_col] = df[product_list_col].apply(ast.literal_eval)
        except:
            pass
    
    # Hitung jumlah item per transaksi (Monetary proxy)
    def count_items(x):
        if isinstance(x, list):
            return len(x)
        return 0
    
    df['item_count'] = df[product_list_col].apply(count_items)
    
    # Jika day_col tidak ada atau sama dengan ID Transaksi, gunakan ID Transaksi sebagai proxy
    # (Asumsi: BASKET_ID yang lebih tinggi = transaksi lebih baru)
    if day_col not in df.columns or day_col == key_col:
        # Cari kolom yang bisa digunakan sebagai proxy waktu
        possible_time_cols = ['ID Transaksi', 'BASKET_ID', 'transaction_id', 'order_id']
        time_col = None
        for col in possible_time_cols:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            # Jika tidak ada, buat index sebagai proxy
            df['_transaction_order'] = range(len(df))
            time_col = '_transaction_order'
    else:
        time_col = day_col
    
    # Tentukan "hari terakhir" dalam dataset (untuk menghitung Recency)
    max_day = df[time_col].max()
    
    # Agregasi per pelanggan
    rfm = df.groupby(key_col).agg({
        time_col: 'max',           # Transaksi terakhir (untuk Recency)
        'item_count': 'sum',       # Total item dibeli (Monetary proxy)
    }).reset_index()
    
    # Hitung Frequency terpisah (jumlah transaksi unik)
    freq = df.groupby(key_col).size().reset_index(name='Frequency')
    rfm = rfm.merge(freq, on=key_col)
    
    # Rename kolom
    rfm.columns = [key_col, 'LastPurchaseDay', 'Monetary', 'Frequency']
    
    # Hitung Recency (semakin kecil = semakin baru = semakin baik)
    # Normalize ke skala yang lebih masuk akal
    rfm['Recency'] = max_day - rfm['LastPurchaseDay']
    
    # Normalize Recency ke range 0-100 untuk interpretasi lebih mudah
    if rfm['Recency'].max() > 0:
        rfm['Recency_Normalized'] = (rfm['Recency'] / rfm['Recency'].max() * 100).round(0)
    else:
        rfm['Recency_Normalized'] = 0
    
    # --- SCORING (1-5, menggunakan quintiles) ---
    # Recency: Skor 5 = Baru belanja (nilai recency kecil)
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    except ValueError:
        # Jika tidak bisa dibagi 5, gunakan cut dengan bins
        rfm['R_Score'] = pd.cut(rfm['Recency'], bins=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    
    # Frequency: Skor 5 = Sering belanja (nilai frequency tinggi)
    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    except ValueError:
        rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Monetary: Skor 5 = Banyak belanja (nilai monetary tinggi)
    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    except ValueError:
        rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Handle NaN scores
    rfm['R_Score'] = rfm['R_Score'].fillna(3).astype(int)
    rfm['F_Score'] = rfm['F_Score'].fillna(3).astype(int)
    rfm['M_Score'] = rfm['M_Score'].fillna(3).astype(int)
    
    # RFM Score gabungan (string untuk segmentasi)
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Total Score (untuk ranking)
    rfm['Total_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    # --- SEGMENTASI PELANGGAN ---
    rfm['Segment'] = rfm.apply(assign_rfm_segment, axis=1)
    
    # Gunakan Recency_Normalized untuk tampilan
    rfm['Recency'] = rfm['Recency_Normalized']
    rfm.drop(columns=['Recency_Normalized', 'LastPurchaseDay'], inplace=True, errors='ignore')
    
    return rfm

def assign_rfm_segment(row):
    """
    Menetapkan segmen pelanggan berdasarkan skor RFM.
    Menggunakan logika bisnis standar retail.
    """
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    
    # Champions: Baru belanja, sering, dan banyak
    if r >= 4 and f >= 4 and m >= 4:
        return '🏆 Champions'
    
    # Loyal Customers: Sering belanja dengan nilai bagus
    if f >= 4 and m >= 3:
        return '💎 Loyal Customers'
    
    # Potential Loyalist: Baru dan lumayan sering
    if r >= 4 and f >= 2 and f <= 4:
        return '🌟 Potential Loyalist'
    
    # New Customers: Baru sekali/dua kali belanja
    if r >= 4 and f <= 2:
        return '🆕 New Customers'
    
    # Promising: Baru tapi belanja sedikit
    if r >= 3 and f <= 2 and m <= 2:
        return '🔮 Promising'
    
    # Need Attention: Dulunya bagus, sekarang menurun
    if r >= 2 and r <= 3 and f >= 2 and f <= 3 and m >= 2 and m <= 3:
        return '⚠️ Need Attention'
    
    # About to Sleep: Mulai jarang belanja
    if r <= 2 and f >= 2 and f <= 3:
        return '😴 About to Sleep'
    
    # At Risk: Dulunya bagus, sekarang jarang
    if r <= 2 and f >= 4:
        return '🚨 At Risk'
    
    # Hibernating: Sudah lama tidak belanja
    if r <= 2 and f <= 2:
        return '❄️ Hibernating'
    
    # Can't Lose Them: Pernah jadi pelanggan terbaik
    if r <= 2 and f >= 4 and m >= 4:
        return '🔥 Can\'t Lose Them'
    
    # Default
    return '📊 Others'

def get_segment_recommendations():
    """
    Mengembalikan rekomendasi aksi untuk setiap segmen RFM.
    """
    return {
        '🏆 Champions': {
            'description': 'Pelanggan terbaik! Baru belanja, sering, dan banyak.',
            'action': 'Berikan reward eksklusif, program VIP, ajak jadi brand ambassador.',
            'priority': 1,
            'color': '#10b981'  # Green
        },
        '💎 Loyal Customers': {
            'description': 'Pelanggan setia dengan nilai transaksi konsisten.',
            'action': 'Upsell produk premium, program membership, early access promo.',
            'priority': 2,
            'color': '#3b82f6'  # Blue
        },
        '🌟 Potential Loyalist': {
            'description': 'Pelanggan baru yang menunjukkan potensi loyalitas.',
            'action': 'Nurture dengan welcome series, diskon pembelian ke-3, product education.',
            'priority': 3,
            'color': '#8b5cf6'  # Purple
        },
        '🆕 New Customers': {
            'description': 'Pelanggan baru, baru 1-2x transaksi.',
            'action': 'Onboarding email, first-purchase discount, product recommendation.',
            'priority': 4,
            'color': '#06b6d4'  # Cyan
        },
        '🔮 Promising': {
            'description': 'Pelanggan baru tapi belanja masih sedikit.',
            'action': 'Edukasi produk, bundle deals, free shipping threshold.',
            'priority': 5,
            'color': '#14b8a6'  # Teal
        },
        '⚠️ Need Attention': {
            'description': 'Pelanggan yang performanya mulai menurun.',
            'action': 'Re-engagement campaign, survey kepuasan, special offer.',
            'priority': 6,
            'color': '#f59e0b'  # Amber
        },
        '😴 About to Sleep': {
            'description': 'Pelanggan yang mulai jarang belanja.',
            'action': 'Win-back email, limited time offer, "We miss you" campaign.',
            'priority': 7,
            'color': '#f97316'  # Orange
        },
        '🚨 At Risk': {
            'description': 'Dulunya pelanggan aktif, sekarang jarang.',
            'action': 'URGENT: Personal outreach, big discount, survey alasan pergi.',
            'priority': 8,
            'color': '#ef4444'  # Red
        },
        '❄️ Hibernating': {
            'description': 'Sudah lama tidak belanja.',
            'action': 'Reactivation campaign agresif atau exclude dari marketing.',
            'priority': 9,
            'color': '#6b7280'  # Gray
        },
        '🔥 Can\'t Lose Them': {
            'description': 'Pernah jadi pelanggan terbaik, sekarang hilang.',
            'action': 'CRITICAL: CEO-level outreach, exclusive comeback offer.',
            'priority': 10,
            'color': '#dc2626'  # Dark Red
        },
        '📊 Others': {
            'description': 'Pelanggan dengan pola tidak terklasifikasi.',
            'action': 'Analisis lebih lanjut, general marketing.',
            'priority': 11,
            'color': '#9ca3af'  # Light Gray
        }
    }