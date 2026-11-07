import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Prediksi Churn Pelanggan", page_icon="📡")

KATEGORI = ["Jenis_Kontrak", "Metode_Bayar", "Layanan_Tambahan"]
ANGKA = ["Lama_Berlangganan", "Tagihan_Bulanan", "Jumlah_Keluhan", "Penggunaan_Data_GB"]


# ------------------------------------------------------------ DATA & MODEL
@st.cache_data
def buat_dataset(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Lama_Berlangganan": rng.integers(1, 73, n),
        "Tagihan_Bulanan": rng.integers(50, 400, n),
        "Jumlah_Keluhan": rng.poisson(1.2, n).clip(0, 10),
        "Penggunaan_Data_GB": rng.integers(1, 120, n),
        "Jenis_Kontrak": rng.choice(["Bulanan", "1 Tahun", "2 Tahun"], n, p=[0.55, 0.25, 0.20]),
        "Metode_Bayar": rng.choice(["Otomatis", "Manual"], n, p=[0.45, 0.55]),
        "Layanan_Tambahan": rng.choice(["Ya", "Tidak"], n, p=[0.4, 0.6]),
    })
    # aturan "tersembunyi" yang membuat peluang churn masuk akal
    skor = (
        -0.04 * df["Lama_Berlangganan"]
        + 0.006 * df["Tagihan_Bulanan"]
        + 0.55 * df["Jumlah_Keluhan"]
        + np.where(df["Jenis_Kontrak"] == "Bulanan", 1.2, np.where(df["Jenis_Kontrak"] == "1 Tahun", 0.0, -1.0))
        + np.where(df["Metode_Bayar"] == "Manual", 0.5, -0.2)
        + np.where(df["Layanan_Tambahan"] == "Ya", -0.5, 0.3)
        - 1.8
    )
    peluang = 1 / (1 + np.exp(-skor))
    df["Churn"] = (rng.random(n) < peluang).astype(int)
    return df


@st.cache_resource
def latih_model():
    df = buat_dataset()
    X, y = df.drop(columns="Churn"), df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    prep = ColumnTransformer([("kat", OneHotEncoder(handle_unknown="ignore"), KATEGORI)], remainder="passthrough")
    model = Pipeline([
        ("prep", prep),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ])
    model.fit(X_tr, y_tr)
    akurasi = accuracy_score(y_te, model.predict(X_te))
    return model, akurasi, len(df), y.mean()


model, akurasi, jumlah_data, rasio_churn = latih_model()

# ------------------------------------------------------------ TAMPILAN
st.title("📡 Prediksi Churn Pelanggan")
st.write("Isi data pelanggan di bawah, lalu klik tombol untuk memprediksi apakah pelanggan **berhenti berlangganan (churn)**.")
st.caption(f"Model: Random Forest · {jumlah_data} data latih · akurasi uji {akurasi * 100:.1f}%")

c1, c2 = st.columns(2)
with c1:
    lama = st.number_input("Lama Berlangganan (bulan)", 1, 72, 12)
    kontrak = st.selectbox("Jenis Kontrak", ["Bulanan", "1 Tahun", "2 Tahun"])
    keluhan = st.slider("Jumlah Keluhan (6 bulan terakhir)", 0, 10, 1)
    bayar = st.selectbox("Metode Pembayaran", ["Otomatis", "Manual"])
with c2:
    tagihan = st.slider("Tagihan Bulanan (ribu Rp)", 50, 400, 150)
    data_gb = st.number_input("Penggunaan Data per Bulan (GB)", 1, 120, 30)
    tambahan = st.selectbox("Berlangganan Layanan Tambahan", ["Ya", "Tidak"])

if st.button("🔮 Prediksi Churn", type="primary", width="stretch"):
    baru = pd.DataFrame([{
        "Lama_Berlangganan": lama,
        "Tagihan_Bulanan": tagihan,
        "Jumlah_Keluhan": keluhan,
        "Penggunaan_Data_GB": data_gb,
        "Jenis_Kontrak": kontrak,
        "Metode_Bayar": bayar,
        "Layanan_Tambahan": tambahan,
    }])
    p = model.predict_proba(baru)[0][1]
    if p >= 0.5:
        st.error(f"⚠️ Pelanggan **berpotensi CHURN** (peluang {p * 100:.1f}%)")
        st.write("Saran: tawarkan promo retensi atau upgrade ke kontrak yang lebih panjang.")
    else:
        st.success(f"✅ Pelanggan **kemungkinan TETAP berlangganan** (peluang churn {p * 100:.1f}%)")
    st.progress(float(p), text=f"Peluang churn: {p * 100:.1f}%")

with st.expander("Lihat data latih"):
    st.dataframe(buat_dataset().head(50), width="stretch")
    st.caption(f"Proporsi churn pada data: {rasio_churn * 100:.1f}%. Data dibuat secara simulasi untuk keperluan belajar data mining.")
