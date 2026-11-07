import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentasi Pelanggan - K-Means", page_icon="🛍️", layout="wide")


# ---------------------------------------------------------------- DATA
@st.cache_data
def buat_data_contoh(n=400, seed=42):
    rng = np.random.default_rng(seed)
    # (usia, pendapatan juta/thn, skor belanja, kunjungan/bln, rata2 transaksi ribu)
    grup = [
        (24, 5, 45, 85, 9, 150, 0.30),   # muda, hemat tapi sering belanja
        (38, 12, 90, 80, 6, 450, 0.20),  # mapan, boros
        (45, 11, 85, 20, 2, 300, 0.20),  # mapan, hemat
        (30, 4, 35, 25, 3, 100, 0.15),   # pemula
        (55, 7, 55, 50, 4, 250, 0.15),   # senior
    ]
    baris = []
    for usia, inc, _, skor, kunj, trx, p in grup:
        m = int(n * p)
        baris.append(pd.DataFrame({
            "Usia": rng.normal(usia, 4, m).clip(18, 70).round(),
            "Pendapatan_Juta": rng.normal(inc * 10, 6, m).clip(10, 200).round(1),
            "Skor_Belanja": rng.normal(skor, 8, m).clip(1, 100).round(),
            "Kunjungan_PerBulan": rng.normal(kunj, 1.5, m).clip(1, 20).round(),
            "Rata2_Transaksi_Ribu": rng.normal(trx, 40, m).clip(20, 1000).round(),
        }))
    df = pd.concat(baris, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "ID_Pelanggan", [f"P{str(i + 1).zfill(4)}" for i in range(len(df))])
    return df


@st.cache_data
def baca_csv(file):
    return pd.read_csv(file)


# ---------------------------------------------------------------- SIDEBAR
st.sidebar.title("⚙️ Pengaturan")
sumber = st.sidebar.radio("Sumber data", ["Dataset contoh (pelanggan toko)", "Upload CSV sendiri"])

if sumber.startswith("Dataset"):
    df = buat_data_contoh()
else:
    f = st.sidebar.file_uploader("Upload file CSV", type="csv")
    if f is None:
        st.info("Upload file CSV di sidebar, atau pilih dataset contoh.")
        st.stop()
    df = baca_csv(f)

kolom_num = df.select_dtypes(include="number").columns.tolist()
if len(kolom_num) < 2:
    st.error("Dataset harus punya minimal 2 kolom numerik.")
    st.stop()

fitur = st.sidebar.multiselect("Fitur untuk clustering", kolom_num, default=kolom_num[: min(4, len(kolom_num))])
if len(fitur) < 2:
    st.warning("Pilih minimal 2 fitur.")
    st.stop()

normalisasi = st.sidebar.checkbox("Standarisasi data (disarankan)", value=True)
k = st.sidebar.slider("Jumlah cluster (K)", 2, 10, 5)
k_maks = st.sidebar.slider("K maksimum untuk Elbow", 4, 12, 10)

# ---------------------------------------------------------------- PREPROCESS
data = df.copy()
data[fitur] = data[fitur].fillna(data[fitur].median())
X = data[fitur].values
scaler = StandardScaler() if normalisasi else None
Xs = scaler.fit_transform(X) if scaler else X


@st.cache_data
def hitung_elbow(Xs, k_maks):
    hasil = []
    for i in range(2, k_maks + 1):
        km = KMeans(n_clusters=i, n_init=10, random_state=42).fit(Xs)
        hasil.append((i, km.inertia_, silhouette_score(Xs, km.labels_)))
    return pd.DataFrame(hasil, columns=["K", "Inertia", "Silhouette"])


model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
data["Cluster"] = model.labels_.astype(str)
sil = silhouette_score(Xs, model.labels_)

# ---------------------------------------------------------------- HEADER
st.title("🛍️ Segmentasi Pelanggan dengan K-Means")
st.caption("Aplikasi data mining: mengelompokkan pelanggan berdasarkan perilaku belanja secara otomatis (unsupervised learning).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah data", len(data))
c2.metric("Jumlah fitur", len(fitur))
c3.metric("Jumlah cluster", k)
c4.metric("Silhouette score", f"{sil:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data & EDA", "📉 Elbow & Silhouette", "🎯 Hasil Cluster", "🧾 Profil Segmen", "🔮 Prediksi Pelanggan Baru"]
)

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.subheader("Cuplikan data")
    st.dataframe(df.head(20), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.subheader("Statistik deskriptif")
        st.dataframe(df[fitur].describe().T.round(2), use_container_width=True)
    with b:
        st.subheader("Korelasi antar fitur")
        fig = px.imshow(df[fitur].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    pilih = st.selectbox("Distribusi fitur", fitur)
    st.plotly_chart(px.histogram(df, x=pilih, nbins=30, marginal="box"), use_container_width=True)

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Menentukan jumlah cluster terbaik")
    ev = hitung_elbow(Xs, k_maks)
    a, b = st.columns(2)
    with a:
        fig = px.line(ev, x="K", y="Inertia", markers=True, title="Metode Elbow (Inertia)")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.line(ev, x="K", y="Silhouette", markers=True, title="Silhouette Score")
        st.plotly_chart(fig, use_container_width=True)
    best = int(ev.loc[ev["Silhouette"].idxmax(), "K"])
    st.success(f"K dengan silhouette tertinggi: **{best}** (skor {ev['Silhouette'].max():.3f}). Kamu sekarang memakai K = {k}.")
    st.dataframe(ev.round(3), use_container_width=True)

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Visualisasi cluster")
    mode = st.radio("Tampilan", ["PCA 2D", "Pilih 2 fitur"], horizontal=True)
    if mode == "PCA 2D":
        pca = PCA(n_components=2, random_state=42)
        P = pca.fit_transform(Xs)
        dfp = pd.DataFrame(P, columns=["PC1", "PC2"])
        dfp["Cluster"] = data["Cluster"]
        fig = px.scatter(dfp, x="PC1", y="PC2", color="Cluster", opacity=0.75,
                         category_orders={"Cluster": sorted(data["Cluster"].unique())})
        cen = pca.transform(model.cluster_centers_)
        fig.add_trace(go.Scatter(x=cen[:, 0], y=cen[:, 1], mode="markers+text",
                                 marker=dict(symbol="x", size=16, color="black"),
                                 text=[f"C{i}" for i in range(k)], textposition="top center", name="Centroid"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Varians terjelaskan PC1 + PC2: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
    else:
        x_ax = st.selectbox("Sumbu X", fitur, index=0)
        y_ax = st.selectbox("Sumbu Y", fitur, index=1)
        fig = px.scatter(data, x=x_ax, y=y_ax, color="Cluster", opacity=0.75,
                         category_orders={"Cluster": sorted(data["Cluster"].unique())})
        st.plotly_chart(fig, use_container_width=True)

    ukuran = data["Cluster"].value_counts().sort_index().reset_index()
    ukuran.columns = ["Cluster", "Jumlah"]
    st.plotly_chart(px.pie(ukuran, names="Cluster", values="Jumlah", title="Proporsi anggota tiap cluster"),
                    use_container_width=True)

# ---------------------------------------------------------------- TAB 4
with tab4:
    st.subheader("Rata-rata fitur per segmen")
    profil = data.groupby("Cluster")[fitur].mean().round(2)
    profil["Jumlah"] = data["Cluster"].value_counts().sort_index()
    st.dataframe(profil, use_container_width=True)

    # Radar chart berdasarkan nilai ternormalisasi 0-1
    mm = (profil[fitur] - profil[fitur].min()) / (profil[fitur].max() - profil[fitur].min()).replace(0, 1)
    fig = go.Figure()
    for c in mm.index:
        fig.add_trace(go.Scatterpolar(r=mm.loc[c].tolist() + [mm.loc[c].iloc[0]],
                                      theta=fitur + [fitur[0]], fill="toself", name=f"Cluster {c}"))
    fig.update_layout(title="Radar profil segmen (skala 0-1)", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretasi otomatis")
    rata = data[fitur].mean()
    for c in profil.index:
        tinggi = [f for f in fitur if profil.loc[c, f] > rata[f] * 1.15]
        rendah = [f for f in fitur if profil.loc[c, f] < rata[f] * 0.85]
        teks = f"**Cluster {c}** ({int(profil.loc[c, 'Jumlah'])} pelanggan): "
        teks += ("tinggi di " + ", ".join(tinggi) + ". ") if tinggi else ""
        teks += ("Rendah di " + ", ".join(rendah) + ".") if rendah else ""
        if not tinggi and not rendah:
            teks += "karakteristik mendekati rata-rata."
        st.markdown("- " + teks)

    st.download_button("⬇️ Download hasil clustering (CSV)", data.to_csv(index=False).encode("utf-8"),
                       "hasil_clustering.csv", "text/csv")

# ---------------------------------------------------------------- TAB 5
with tab5:
    st.subheader("Masukkan data pelanggan baru")
    cols = st.columns(min(3, len(fitur)))
    nilai = {}
    for i, f in enumerate(fitur):
        mn, mx, md = float(df[f].min()), float(df[f].max()), float(df[f].median())
        nilai[f] = cols[i % len(cols)].number_input(f, min_value=mn, max_value=mx, value=md)
    if st.button("Prediksi segmen", type="primary"):
        baru = np.array([[nilai[f] for f in fitur]])
        baru_s = scaler.transform(baru) if scaler else baru
        c = int(model.predict(baru_s)[0])
        jarak = np.linalg.norm(model.cluster_centers_ - baru_s, axis=1)
        st.success(f"Pelanggan ini masuk **Cluster {c}**")
        st.write("Jarak ke tiap centroid:")
        st.dataframe(pd.DataFrame({"Cluster": range(k), "Jarak": jarak.round(3)}).set_index("Cluster").T,
                     use_container_width=True)
