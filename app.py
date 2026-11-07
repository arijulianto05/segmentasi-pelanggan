import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Prediksi Churn Pelanggan IndiHome", page_icon="📶")

KATEGORI = ["Layanan_Tambahan", "Metode_Bayar"]
PAKET = [10, 20, 30, 50, 100, 200]


# ------------------------------------------------------------ DATA & MODEL
@st.cache_data
def buat_dataset(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Lama_Berlangganan": rng.integers(1, 73, n),
        "Paket_Mbps": rng.choice(PAKET, n, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]),
        "Gangguan_Per_Bulan": rng.poisson(1.0, n).clip(0, 10),
        "Jumlah_Keluhan": rng.poisson(1.0, n).clip(0, 10),
        "Telat_Bayar_6Bulan": rng.poisson(0.8, n).clip(0, 6),
        "Layanan_Tambahan": rng.choice(["IndiHome TV", "Internet Saja"], n, p=[0.45, 0.55]),
        "Metode_Bayar": rng.choice(["Autodebet", "Bayar Manual"], n, p=[0.4, 0.6]),
    })
    # aturan "tersembunyi" yang membuat peluang churn masuk akal
    skor = (
        -0.035 * df["Lama_Berlangganan"]
        + 0.5 * df["Gangguan_Per_Bulan"]
        + 0.4 * df["Jumlah_Keluhan"]
        + 0.35 * df["Telat_Bayar_6Bulan"]
        + np.where(df["Paket_Mbps"] <= 20, 0.4, np.where(df["Paket_Mbps"] >= 100, -0.3, 0.0))
        + np.where(df["Layanan_Tambahan"] == "IndiHome TV", -0.5, 0.3)
        + np.where(df["Metode_Bayar"] == "Bayar Manual", 0.5, -0.3)
        - 1.5
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
st.title("📶 Prediksi Churn Pelanggan IndiHome")
st.write("Isi data pelanggan di bawah, lalu klik tombol untuk memprediksi apakah pelanggan akan **berhenti berlangganan (churn)**.")
st.caption(f"Model: Random Forest · {jumlah_data} data latih · akurasi uji {akurasi * 100:.1f}%")

c1, c2 = st.columns(2)
with c1:
    lama = st.number_input("Lama Berlangganan (bulan)", 1, 72, 12)
    paket = st.selectbox("Paket Internet (Mbps)", PAKET, index=2)
    gangguan = st.slider("Gangguan Jaringan per Bulan (kali)", 0, 10, 1)
    layanan = st.selectbox("Layanan Tambahan", ["IndiHome TV", "Internet Saja"])
with c2:
    keluhan = st.slider("Jumlah Keluhan (6 bulan terakhir)", 0, 10, 1)
    telat = st.slider("Telat Bayar (6 bulan terakhir, kali)", 0, 6, 0)
    bayar = st.selectbox("Metode Pembayaran", ["Autodebet", "Bayar Manual"])

if st.button("Prediksi Churn", type="primary", width="stretch"):
    baru = pd.DataFrame([{
        "Lama_Berlangganan": lama,
        "Paket_Mbps": paket,
        "Gangguan_Per_Bulan": gangguan,
        "Jumlah_Keluhan": keluhan,
        "Telat_Bayar_6Bulan": telat,
        "Layanan_Tambahan": layanan,
        "Metode_Bayar": bayar,
    }])
    p = model.predict_proba(baru)[0][1]
    if p >= 0.5:
        st.error(f"⚠️ Pelanggan **berpotensi CHURN** (peluang {p * 100:.1f}%)")
        saran = []
        if gangguan >= 3:
            saran.append("prioritaskan perbaikan jaringan di lokasi pelanggan")
        if keluhan >= 3:
            saran.append("hubungi pelanggan untuk menyelesaikan keluhan")
        if bayar == "Bayar Manual":
            saran.append("tawarkan autodebet")
        if layanan == "Internet Saja":
            saran.append("tawarkan paket bundling dengan IndiHome TV")
        if not saran:
            saran.append("berikan promo retensi atau penawaran upgrade paket")
        st.write("Saran tindak lanjut: " + "; ".join(saran) + ".")
    else:
        st.success(f"✅ Pelanggan **kemungkinan TETAP berlangganan** (peluang churn {p * 100:.1f}%)")
    st.progress(float(p), text=f"Peluang churn: {p * 100:.1f}%")

with st.expander("Lihat data latih"):
    st.dataframe(buat_dataset().head(50), width="stretch")
    st.caption(
        f"Proporsi churn pada data: {rasio_churn * 100:.1f}%. "
        "Data dibuat secara simulasi untuk keperluan belajar data mining, "
        "bukan data resmi IndiHome/Telkom."
    )
