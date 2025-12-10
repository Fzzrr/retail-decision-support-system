import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

@st.cache_data
def split_and_resample(X, y, method='undersampling'):
    """
    Melakukan split data dan resampling (hanya pada data training).
    Aman untuk:
      - Kelas minoritas sangat sedikit (SMOTE bisa error)
      - Fitur hasil one-hot encoding yang bertipe boolean
    """

    # 🔹 Pastikan semua fitur numerik (SMOTE tidak bisa untuk boolean)
    # Jika sebelumnya hasil encode berupa bool, ini akan mengubahnya ke 0.0 / 1.0
    X = X.astype(float)

    # 1. Split data menjadi train dan test (Test set asli, imbalanced)
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 🔹 Metode Random Under-Sampling
    if method == 'undersampling':
        sampler = RandomUnderSampler(random_state=42)
        st.write("Menerapkan Random Under-Sampling (RUS) pada data latih...")
        X_train_res, y_train_res = sampler.fit_resample(X_train_orig, y_train_orig)
        return X_train_res, y_train_res, X_test, y_test

    # 🔹 Metode SMOTE (Oversampling)
    elif method == 'oversampling':
        st.write("Mencoba menerapkan SMOTE (Over-sampling) pada data latih...")

        counter = Counter(y_train_orig)
        st.write("Distribusi y_train sebelum SMOTE:", dict(counter))

        # Jika hanya ada satu kelas di y_train, SMOTE tidak bisa dipakai
        if len(counter) < 2:
            st.warning(
                "⚠️ Data latih hanya memiliki satu kelas. "
                "SMOTE tidak bisa dijalankan. Model akan dilatih tanpa resampling."
            )
            return X_train_orig, y_train_orig, X_test, y_test

        # Cari kelas minoritas & jumlahnya
        minority_class = min(counter, key=counter.get)
        minority_count = counter[minority_class]

        # Minimal 2 sampel di kelas minoritas untuk SMOTE
        if minority_count < 2:
            st.warning(
                f"⚠️ Jumlah sampel kelas minoritas di data latih sangat sedikit ({minority_count}). "
                "SMOTE tidak dijalankan. Model akan dilatih tanpa resampling."
            )
            return X_train_orig, y_train_orig, X_test, y_test

        # Atur k_neighbors supaya tidak melebihi (minority_count - 1)
        k_neighbors = min(5, minority_count - 1)
        if k_neighbors < 1:
            k_neighbors = 1

        st.write(
            f"✅ Menjalankan SMOTE dengan k_neighbors={k_neighbors} "
            f"(kelas minoritas={minority_class}, jumlah={minority_count})"
        )

        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = sampler.fit_resample(X_train_orig, y_train_orig)
        return X_train_res, y_train_res, X_test, y_test

    # 🔹 Jika method tidak dikenali → tidak ada resampling
    else:
        st.write("Metode resampling tidak dikenali. Tidak ada resampling yang diterapkan.")
        return X_train_orig, y_train_orig, X_test, y_test

@st.cache_resource
def train_ann_model(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(14,),
        activation='tanh',
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    return model

def get_predictions(_model, X_data):
    X_data_reordered = X_data.reindex(columns=_model.feature_names_in_, fill_value=0)
    probs = _model.predict_proba(X_data_reordered)[:, 1]
    preds = _model.predict(X_data_reordered)
    return probs, preds

def generate_evaluation_metrics(_model, X_test, y_test):
    probs, preds = get_predictions(_model, X_test)
    results = {}

    try:
        results['report'] = classification_report(y_test, preds, output_dict=True)
    except Exception:
        results['report'] = {"error": "Gagal membuat report."}

    try:
        results['auc'] = roc_auc_score(y_test, probs)
    except Exception:
        results['auc'] = 0.0

    # --- PERBAIKAN UKURAN GAMBAR (FIXED SIZE) ---
    plot_size = (6, 5) # Lebar 6, Tinggi 5 (Seragamkan!)

    # 1. Plot Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=plot_size)
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax_cm, cmap=plt.cm.Blues, colorbar=True)
    ax_cm.set_title("Confusion Matrix (Test Set)")
    # Agar layout pas dan tidak terpotong
    fig_cm.tight_layout()
    results['cm_plot'] = fig_cm

    # 2. Plot ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=plot_size)
    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"AUC = {results['auc']:.4f}", color='#2e7bcf', linewidth=2)
    except Exception:
        ax_roc.text(0.5, 0.5, "Gagal ROC", ha='center')

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve (Test Set)')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    # Agar layout pas
    fig_roc.tight_layout() 
    results['roc_plot'] = fig_roc

    return results