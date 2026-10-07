from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Dataset
# Setiap baris = satu data
# Kolom 1 = jam belajar
# Kolom 2 = nilai ujian

X = [
    [1, 40],
    [2, 45],
    [1, 50],

    [5, 70],
    [6, 75],
    [5, 80],

    [8, 90],
    [9, 92],
    [10, 95],
]


# 2. Membuat model K-Means
# Kita ingin mencari 3 kelompok
model = KMeans(
    n_clusters=3,
    random_state=47,
    n_init='auto',
)


# 3. Training / mencari cluster
model.fit(X)

 
# 4. Melihat cluster setiap data
labels = model.labels_

print("Cluster setiap data:")
print(labels)


# 5. Melihat posisi centroid
print("\nCentroid:")
print(model.cluster_centers_)


# 6. Visualisasi hasil clustering
plt.scatter(
    [data[0] for data in X],
    [data[1] for data in X],
    c=labels,
)

# Menampilkan centroid
centroids = model.cluster_centers_

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
)

plt.xlabel("Jam Belajar")
plt.ylabel("Nilai Ujian")
plt.title("K-Means Clustering")

plt.savefig('images/ch04_01.png', dpi=150, bbox_inches='tight')
