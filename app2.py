# ============================================================
# STREAMLIT DASHBOARD: TIME SERIES CLUSTERING
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tslearn.metrics import dtw
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(layout="wide")
st.title("📊 Time Series Clustering Dashboard")

# ============================================================
# LOAD DATA
# ============================================================

FILE_PATH = "SuperStore_Sales_Updated.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(FILE_PATH)
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

df = load_data()

# ============================================================
# PREPROCESSING
# ============================================================

df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month

ts_data = (df.groupby(['year','month','category','region'])['sales']
             .sum().reset_index())

ts_data['period'] = ts_data['year'].astype(str) + '-' + ts_data['month'].astype(str).str.zfill(2)

ts_pivot = ts_data.pivot_table(
    index='period',
    columns=['category','region'],
    values='sales'
).fillna(0)

ts_pivot.columns = ['_'.join(col) for col in ts_pivot.columns]

ts_norm = ts_pivot.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# ============================================================
# STATISTIK DESKRIPTIF (WAJIB TAMBAH)
# ============================================================

def hitung_ce(series):
    return np.sqrt(np.sum(np.diff(series) ** 2))

stats_dict = {}

for col in ts_pivot.columns:
    s = ts_pivot[col].values
    
    stats_dict[col] = {
        'Mean'       : np.mean(s),
        'Std'        : np.std(s),
        'CV'         : np.std(s) / np.mean(s) if np.mean(s) != 0 else 0,
        'Min'        : np.min(s),
        'Max'        : np.max(s),
        'Slope_Tren' : np.polyfit(range(len(s)), s, 1)[0],
        'ACF_lag1'   : pd.Series(s).autocorr(lag=1),
        'ACF_lag2'   : pd.Series(s).autocorr(lag=2),
        'CE'         : hitung_ce(s)
    }

stats_df = pd.DataFrame(stats_dict).T
stats_df.index.name = "Series"

# TRANSFORM
ts_matrix = ts_norm.T.values
series_names = ts_norm.columns.tolist()

# ============================================================
# DISTANCE FUNCTIONS
# ============================================================

def complexity(series):
    return np.sqrt(np.sum(np.diff(series)**2))

def dtw_matrix(data):
    n = len(data)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = dtw(data[i], data[j])
    return D

def cid_dtw_matrix(data):
    n = len(data)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            d = dtw(data[i], data[j])
            ce_i = complexity(data[i])
            ce_j = complexity(data[j])
            cf = max(ce_i, ce_j) / min(ce_i, ce_j)
            D[i,j] = d * cf
    return D

# ============================================================
# HITUNG DISTANCE
# ============================================================

D_dtw = dtw_matrix(ts_matrix)
D_cid = cid_dtw_matrix(ts_matrix)

# ============================================================
# CLUSTERING
# ============================================================

Z_dtw = linkage(squareform(D_dtw), method='ward')
Z_cid = linkage(squareform(D_cid), method='ward')

k_optimal = 5

k = st.sidebar.slider("Jumlah Cluster (k)", 2, 5, k_optimal)

st.sidebar.markdown(f"**📌 k optimal (hasil analisis): {k_optimal}**")

cluster_dtw = fcluster(Z_dtw, k, criterion='maxclust')
cluster_cid = fcluster(Z_cid, k, criterion='maxclust')

# ============================================================
# FUNGSI VISUALISASI CLUSTER
# ============================================================

def plot_cluster(data, labels, title):
    fig, ax = plt.subplots(figsize=(10,5))
    
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        
        # plot anggota cluster
        for i in idx:
            ax.plot(data[i], alpha=0.2)
        
        # centroid
        centroid = np.mean(data[idx], axis=0)
        ax.plot(centroid, linewidth=3, label=f'Cluster {c}')
    
    ax.set_title(title)
    ax.grid()
    ax.legend()
    
    return fig

def plot_raw_timeseries(ts_pivot):

    categories = ['Furniture', 'Office Supplies', 'Technology']
    regions    = ['Central', 'East', 'South', 'West']
    cat_colors = {'Furniture': '#4C72B0',
                  'Office Supplies': '#DD8452',
                  'Technology': '#55A868'}

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=False)
    fig.suptitle("Sales Bulanan per Kategori × Region", fontsize=14)

    x_labels = list(ts_pivot.index)
    x_idx    = range(len(x_labels))

    for i, cat in enumerate(categories):
        for j, reg in enumerate(regions):
            ax  = axes[i][j]
            col = f"{cat}_{reg}"
            y   = ts_pivot[col].values / 1000

            ax.plot(x_idx, y, color=cat_colors[cat], linewidth=2, marker='o')
            ax.fill_between(x_idx, y, alpha=0.12, color=cat_colors[cat])
            ax.set_title(f"{cat[:4]} - {reg}", fontsize=9)
            ax.set_xticks(x_idx[::6])
            ax.set_xticklabels([x_labels[k][2:] for k in x_idx[::6]], rotation=30)

    plt.tight_layout()
    return fig


def plot_cv_ce(stats_df):

    categories = ['Furniture', 'Office Supplies', 'Technology']
    regions    = ['Central', 'East', 'South', 'West']

    cv_matrix = stats_df['CV'].values.reshape(3, 4)
    ce_matrix = (stats_df['CE'].values / 1000).reshape(3, 4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(cv_matrix, ax=axes[0], annot=True, cmap='YlOrRd')
    axes[0].set_title("CV")

    sns.heatmap(ce_matrix, ax=axes[1], annot=True, cmap='Blues')
    axes[1].set_title("CE")

    return fig

def plot_slope(stats_df):

    slope_df = stats_df[['Slope_Tren']].copy().reset_index()
    slope_df = slope_df.sort_values('Slope_Tren')

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(slope_df['Series'], slope_df['Slope_Tren'])
    ax.set_title("Slope Tren")

    return fig

def plot_normalized(ts_norm):

    fig, ax = plt.subplots(figsize=(10,5))

    for col in ts_norm.columns:
        ax.plot(ts_norm[col].values, alpha=0.4)

    ax.set_title("Time Series (Z-score)")
    return fig

# ============================================================
# FUNGSI VISUALISASI CLUSTER
# ============================================================
k_final = k
labels_final = cluster_cid

# warna cluster
CLUSTER_COLORS = {
    1: '#4C72B0',
    2: '#DD8452',
    3: '#55A868',
    4: '#C44E52',
    5: '#8172B2'
}

# nama pendek
short_names = series_names

# label deskriptif sederhana
cluster_labels_desc = {i: f"Cluster {i}" for i in range(1, k_final+1)}
def plot_mean_trajectory(ts_norm, labels_final, k_final):

    periods = list(ts_norm.index)
    x_idx = range(len(periods))

    fig, axes = plt.subplots(1, k_final, figsize=(5*k_final, 4))
    if k_final == 1:
        axes = [axes]

    for ax, k_id in zip(axes, range(1, k_final+1)):
        sub_idx = np.where(labels_final == k_id)[0]
        color = CLUSTER_COLORS[k_id]

        for idx in sub_idx:
            ax.plot(ts_norm.iloc[:, idx].values, alpha=0.2, color=color)

        mean_traj = ts_norm.iloc[:, sub_idx].mean(axis=1)
        ax.plot(mean_traj, color=color, linewidth=3)

        ax.set_title(f"Cluster {k_id}")
        ax.grid()

    return fig
def build_cluster_profile(ts_pivot, labels_final):
    
    df_list = []
    
    for k_id in np.unique(labels_final):
        idx = np.where(labels_final == k_id)[0]
        
        subset = ts_pivot.iloc[:, idx]
        
        df_list.append({
            'Cluster': k_id,
            'Mean_Sales': subset.mean().mean(),
            'CV_mean': subset.std().mean() / subset.mean().mean(),
            'Slope_mean': np.mean([np.polyfit(range(len(subset[col])), subset[col],1)[0] for col in subset]),
            'ACF_lag1_mean': np.mean([pd.Series(subset[col]).autocorr(lag=1) for col in subset]),
            'CE_mean': np.mean([np.sqrt(np.sum(np.diff(subset[col])**2)) for col in subset])
        })
    
    return pd.DataFrame(df_list).set_index('Cluster')
def plot_radar(df_profile):

    features = ['CV_mean','ACF_lag1_mean','Slope_mean','CE_mean','Mean_Sales']
    
    df_norm = (df_profile - df_profile.min()) / (df_profile.max() - df_profile.min())
    
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for k_id in df_norm.index:
        vals = df_norm.loc[k_id].values
        vals = np.concatenate([vals, [vals[0]]])
        
        ax.plot(angles, vals, label=f"Cluster {k_id}")
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.legend()

    return fig
def plot_heatmap_profile(df_profile):

    df_norm = (df_profile - df_profile.min()) / (df_profile.max() - df_profile.min())

    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(df_norm, annot=True, cmap='RdYlGn', ax=ax)

    return fig
def plot_boxplot(ts_pivot, labels_final):

    df = ts_pivot.T.copy()
    df['Cluster'] = labels_final

    melted = df.melt(id_vars='Cluster')

    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=melted, x='Cluster', y='value', ax=ax)

    return fig


# ============================================================
# SIDEBAR MENU
# ============================================================

menu = st.sidebar.selectbox("Navigasi Analisis", [
    "1. Latar Belakang",
    "2. Persiapan Data",
    "3. Normalisasi",
    "4. Eksplorasi & Statistik",
    "5. Justifikasi Metode",
    "6. Distance Analysis",
    "7. Clustering",
    "8. Anggota Cluster",
    "9. Kesimpulan"
])
if menu == "1. Latar Belakang":
    st.header("Latar Belakang")
    
    st.markdown("""
**METODE: CID-DTW + Hierarchical Ward Linkage**

* Dataset: SuperStore Sales (2019-2020)
* Link Akses : https://www.kaggle.com/code/ekajaya/deep-analysis-superstore-sales-dataset/input
                
Data penjualan ritel lintas kategori dan wilayah jarang memiliki pola temporal
yang seragam. Sebagian segmen tumbuh cepat, sebagian stagnan, sebagian sangat
fluktuatif. Untuk mengidentifikasi kelompok segmen yang berperilaku serupa secara
temporal, digunakan Clustering Deret Waktu berbasis CID-DTW - sebuah
ukuran jarak yang menggabungkan fleksibilitas penyelarasan temporal DTW dengan
koreksi faktor kompleksitas CID.


Dataset yang digunakan adalah SuperStore Sales - sebuah dataset transaksi ritel yang 
mencatat 5.901 transaksi di Amerika Serikat selama periode Januari 2019 hingga Desember 2020, 
mencakup tiga kategori produk (Furniture, Office Supplies, Technology) dan empat wilayah (Central, East, South, West). 
Setelah diagregasi secara bulanan per kombinasi kategori-wilayah, diperoleh 12 deret waktu dengan panjang T = 24 periode.
    
    
**Alasan Pemilihan CID-DTW**

Tiga temuan eksplorasi awal yang menentukan pilihan metode:

- **Volatilitas heterogen** — CV rata-rata 0,603 (rentang 0,37–0,75), artinya
  tingkat fluktuasi antar deret sangat berbeda → koreksi CE diperlukan.
- **Pergeseran temporal** — ACF lag-1 berkisar -0,42 hingga +0,73, artinya
  pola serupa bisa muncul di waktu yang berbeda → DTW diperlukan.
- **Kompleksitas berbeda** — CE berkisar 10.574–31.434 (3× lipat), artinya
  tanpa koreksi, deret fluktuatif selalu tampak "berbeda" → CID wajib.

---
**Alur Analisis**

| Tahap | Nama                                 | Isi Utama                                                                            |
| ----- | ------------------------------------ | ------------------------------------------------------------------------------------ |
| I     | Persiapan & Pra-Pemrosesan           | Agregasi bulanan, normalisasi Z-score, eksplorasi CV, CE, ACF, dan slope tren        |
| II    | Perhitungan Matriks Jarak            | Bangun matriks 12×12 menggunakan CID-DTW, DTW, dan Euclidean sebagai pembanding      |
| III   | Hierarchical Clustering + Dendrogram | Ward Linkage, evaluasi Cophenetic Correlation Coefficient (CCC)                      |
| IV    | Validasi Klaster Optimal             | Penetapan k terbaik via Silhouette (SI), Calinski-Harabasz (CH), Davies-Bouldin (DB) |
| V     | Interpretasi & Profiling Klaster     | Mean trajectory, radar chart, uji Kruskal-Wallis                                     |
| VI    | Perbandingan Distance Measure        | CID-DTW vs DTW vs Euclidean vs ACF, skor komposit berbobot                           |
                """)
elif menu == "2. Persiapan Data":
    st.header("Persiapan Data")
    st.write("Jumlah data:", len(df))
    st.write("Kategori:", df['category'].unique())
    st.write("Region:", df['region'].unique())
    st.markdown("""
Berikut disajikan 5 baris pertama data penjualan yang telah diproses menjadi format time 
series bulanan per kategori dan wilayah.
            """)
    st.dataframe(ts_pivot.head())
elif menu == "3. Normalisasi":
    st.header("Normalisasi Z-Score")
    st.markdown("""
Pada bagian ini dilakukan normalisasi Z-score untuk memastikan semua deret waktu 
berada pada skala yang sama, sehingga perhitungan jarak tidak bias terhadap skala asli.
            """)
    st.write("Mean setelah normalisasi:")
    st.write(ts_norm.mean())
    
    st.write("Std setelah normalisasi:")
    st.write(ts_norm.std())
elif menu == "4. Eksplorasi & Statistik":
    st.header("Eksplorasi Data Time Series")

    st.markdown("""
#### **Gambar 1 — Plot 12 Deret Waktu**

Gambar 1 menyajikan plot deret waktu penjualan bulanan seluruh 12 kombinasi
kategori–wilayah dalam format grid 3 × 4 sebelum normalisasi dilakukan. Seluruh
deret menunjukkan pola fluktuatif tanpa tren yang seragam, dengan beberapa deret
memperlihatkan lonjakan tajam pada bulan-bulan tertentu yang kemungkinan
berkaitan dengan siklus promosi akhir tahun. Perbedaan skala absolut antar deret
tampak jelas, yang memperkuat kebutuhan normalisasi sebelum perhitungan jarak
dilakukan.
            """)
    st.subheader("Raw Time Series")
    st.pyplot(plot_raw_timeseries(ts_pivot))

    st.subheader("Statistik Deskriptif")
    st.dataframe(stats_df)

    st.markdown("""
#### **Gambar 2 — Bar Chart Slope Tren**

Gambar 2 menyajikan nilai slope tren linear masing-masing deret dalam bentuk
diagram batang yang diurutkan dari terkecil hingga terbesar. Technology–South
berada di posisi terbawah dengan slope beta = 46,3 USD/bulan,
mencerminkan pertumbuhan yang hampir stagnan. Office Supplies–West mencatat
slope tertinggi sebesar beta = 695,7 USD/bulan. Jarak antara
slope terendah dan tertinggi yang mencapai lebih dari 15 kali lipat secara
visual menggambarkan heterogenitas fase pertumbuhan antar segmen pasar.
            """)
    st.subheader("Slope Tren")
    st.pyplot(plot_slope(stats_df))
elif menu == "5. Justifikasi Metode":
    st.header("Justifikasi CID-DTW")
    st.markdown("""
#### **Gambar Heatmap CV dan CE**

Gambar tersebut terdiri dari dua heatmap berdampingan yang menggambarkan nilai CV dan
CE untuk setiap kombinasi kategori–wilayah. Sel berwarna lebih gelap pada
heatmap CV mengindikasikan volatilitas lebih tinggi, dengan kombinasi Office
Supplies–West menempati nilai tertinggi CV = 0,745 dan Furniture–West
terendah CV = 0,374. Pada heatmap CE, Technology–East tampak paling
kompleks (CE = 31.434), sedangkan Office Supplies–South mencatat CE terendah
(CE = 10.574). Kontras warna yang jelas pada kedua heatmap secara visual
mempertegas heterogenitas ke-12 deret sebagai landasan pemilihan CID-DTW.
            """)
    st.subheader("Heatmap CV & CE")
    st.pyplot(plot_cv_ce(stats_df))

   # st.subheader("Normalized Time Series")
   # st.pyplot(plot_normalized(ts_norm))

    st.markdown("""
    - CV tinggi → data volatil
    - CE berbeda → kompleksitas berbeda
    - Maka CID-DTW diperlukan
    """)
elif menu == "6. Distance Analysis":
    st.header("Perbandingan Distance")
    st.markdown("""
Tahap ini bertujuan menghitung matriks jarak CID-DTW berukuran
12×12 untuk seluruh pasangan deret waktu yang telah dinormalisasi pada Tahap sebelumnya. 
Perhitungan ini mengintegrasikan tiga komponen: jarak Dynamic Time Warping (DTW) sebagai ukuran jarak temporal, 
faktor koreksi kompleksitas (CF) berbasis Complexity Estimate (CE), dan perkalian keduanya menghasilkan CID-DTW. 
Matriks jarak ini menjadi input utama algoritma hierarchical clustering pada tahap berikutnya.
    """)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    
    sns.heatmap(D_dtw, ax=ax[0])
    ax[0].set_title("DTW")
    
    sns.heatmap(D_cid, ax=ax[1])
    ax[1].set_title("CID-DTW")
    
    st.pyplot(fig)

    st.markdown("""
Ringkasan NIlai jarak
| Metrik    | Min    | Max     | Mean   |
| --------- | ------ | ------- | ------ |
| DTW       | 1,4827 | 5,5006  | 3,7557 |
| CID-DTW   | 1,5607 | 12,3725 | 5,6136 |

Pasangan Paling Mirip (CID-DTW terkecil):
| Pasangan                                     | DTW    | CF     | CID-DTW |
| -------------------------------------------- | ------ | ------ | ------- |
| Office Supplies_East ↔ Office Supplies_South | 1,4827 | 1,0526 | 1,5607  |
| Office Supplies_East ↔ Office Supplies_West  | 1,9587 | 1,1729 | 2,2973  |
| Office Supplies_South ↔ Office Supplies_West | 1,9733 | 1,2346 | 2,4363  |
| Furniture_East ↔ Office Supplies_West        | 2,2333 | 1,2610 | 2,8163  |
| Furniture_East ↔ Office Supplies_East        | 2,1083 | 1,4790 | 3,1182  |
                
Pasangan Paling Berbeda (CID-DTW terbesar):
| Pasangan                                  | DTW    | CF     | CID-DTW |
| ----------------------------------------- | ------ | ------ | ------- |
| Office Supplies_South ↔ Technology_South  | 5,4891 | 2,2540 | 12,3725 |
| Office Supplies_East ↔ Technology_South   | 5,5006 | 2,1413 | 11,7784 |
| Furniture_Central ↔ Office Supplies_East  | 4,9539 | 2,2373 | 11,0832 |
| Furniture_Central ↔ Office Supplies_South | 4,6056 | 2,3550 | 10,8464 |
| Office Supplies_South ↔ Technology_West   | 5,0566 | 2,0468 | 10,3498 |
    """)
elif menu == "7. Clustering":

    st.header("Analisis Clustering Lanjutan")

    # build profile
    df_profile = build_cluster_profile(ts_pivot, labels_final)

    st.subheader("Mean Trajectory")
    st.pyplot(plot_mean_trajectory(ts_norm, labels_final, k_final))

    st.subheader("Radar Profil Cluster")
    st.pyplot(plot_radar(df_profile))

    st.subheader("Heatmap Profil")
    st.pyplot(plot_heatmap_profile(df_profile))

    st.subheader("Distribusi Sales")
    st.pyplot(plot_boxplot(ts_pivot, labels_final))
elif menu == "8. Anggota Cluster":
    st.header("Interpretasi Tiap Anggota Cluster")
    
    for c in np.unique(cluster_cid):
        idx = np.where(cluster_cid == c)[0]
        
        st.subheader(f"Cluster {c}")
        
        for i in idx:
            st.write(series_names[i])

    st.markdown("""
---
**Karakteristik Klaster Terbentuk**

| Klaster | Label | Anggota | Mean SI |
|:---:|---|---|:---:|
| **1** | Sales Tinggi & Pertumbuhan Cepat | OfficeSup\_East, OfficeSup\_South, OfficeSup\_West | 0,5340 |
| **2** | Sales Rendah & Pertumbuhan Lambat | Furniture\_Central, Technology\_South | 0,1489 |
| **3** | Sales Menengah & Pertumbuhan Sedang | Technology\_East, Technology\_West | 0,0534 |
| **4** | Sales Menengah & Pertumbuhan Sedang | Furniture\_South, Technology\_Central | 0,1884 |
| **5** | Sales Menengah & Pertumbuhan Sedang | Furniture\_East, Furniture\_West, OfficeSup\_Central | 0,0145 |

Klaster 1 memiliki Mean SI tertinggi (0,5340), mengindikasikan
bahwa deret penjualan Office Supplies di region Timur, Selatan,
dan Barat merupakan kelompok paling **kohesif dan homogen**.
Klaster 5 memiliki Mean SI terendah (0,0145), menunjukkan bahwa anggotanya berada di wilayah batas antar-klaster dan bersifat
paling tidak stabil.
    """)
elif menu == "9. Kesimpulan":
    st.header("Kesimpulan")
    
    st.markdown("""
    - DTW efektif menangkap pola
    - CID-DTW menambahkan dimensi kompleksitas
    - Cluster menunjukkan segmentasi perilaku penjualan
    """)
