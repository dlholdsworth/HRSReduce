from astropy.io import fits 
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import find_peaks
data=fits.getdata("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0312/reduced/MR_Master_Flat_H20230312.fits")

n_col = data.shape[1]
n_row = data.shape[0]
data2 = np.zeros((n_row,n_col))
data_med=np.median(data)

data3 = gaussian_filter(data, sigma=5)

tmp1 = data[:,0]
tmp3 = data3[:,0]
peak_idx,_ = find_peaks(tmp3)

x = []
y = []

for col in range(n_col):
    tmp3 = data3[:,col]
    peak_idx,_ = find_peaks(tmp3)
    for pk in peak_idx:
        x.append(col)
        y.append(pk)

x = np.array(x)
y = np.array(y)

plt.imshow(data,origin='lower',vmin=0,vmax=100)
plt.plot(x,y,'.')
plt.show()

cluster_x = []
cluster_y = []
cluster_no = []
points= []
for i in range(len(x)):
    points.append((x[i],y[i]))
    points.append((x[i],y[i]+1))
    points.append((x[i],y[i]-1))
points= np.array(points)


from sklearn.cluster import DBSCAN


# Sample list of (x, y) points
#points = [(1, 1), (1.2, 1.1), (5, 5), (5.1, 5.2), (10, 10), (10.2, 10.1), (1.3, 1.2)]
points_np = np.array(points)

# Clustering using DBSCAN (Density-Based Spatial Clustering)
# eps: max distance between two samples for them to be in the same cluster
# min_samples: minimum number of points to form a dense region
db = DBSCAN(eps=2, min_samples=3).fit(points_np)
labels = db.labels_  # Cluster labels for each point
print(labels)

# Prepare colors for clusters
unique_labels = set(labels)
#colors = plt.cm.get_cmap("tab10", len(unique_labels))

# Plot clusters and connect points within the same cluster
plt.figure(figsize=(6, 6))

for label in unique_labels:
    print(label)
    cluster_points = points_np[labels == label][0]
    plt.plot(cluster_points[0],cluster_points[1],'.')
#    color = colors(label)
#
#    # Draw all connections within the cluster
#    for i in range(len(cluster_points)):
#        for j in range(i + 1, len(cluster_points)):
#            plt.plot(
#                [cluster_points[i][0], cluster_points[j][0]],
#                [cluster_points[i][1], cluster_points[j][1]],
#                marker='o'
#            )

plt.title("Efficient Clustering and Connection of Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()







from sklearn.cluster import KMeans
from numpy.polynomial.polynomial import Polynomial

# 1. Generate or load your (x, y) points
# For demo, we'll generate points from 3 noisy polynomials
def generate_poly_points(coeffs, n_points, noise_std=0.5):
    x = np.linspace(-10, 10, n_points)
    y = sum(c * x**i for i, c in enumerate(coeffs)) + np.random.normal(0, noise_std, n_points)
    return np.column_stack((x, y))

#points = np.vstack([
#    generate_poly_points([1, -2, 0.5], 100),   # quadratic
#    generate_poly_points([-1, 1, -0.2], 100),  # another
#    generate_poly_points([0, 1.5], 100)        # linear
#])

print(points)

# 2. Cluster into `n` groups using KMeans
n_polynomials = 84
kmeans = KMeans(n_clusters=n_polynomials)
labels = kmeans.fit_predict(points)

# 3. Fit polynomials to each cluster
degree = 4  # Set polynomial degree
colors = plt.cm.get_cmap("tab10", n_polynomials)

plt.figure(figsize=(8, 6))

for i in range(n_polynomials):
    cluster_points = points[labels == i]
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]

    # Fit polynomial (least squares)
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    # Plot points
    plt.scatter(x, y, s=10, color=colors(i), label=f'Cluster {i}')

    # Plot fitted polynomial
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = poly(x_fit)
    plt.plot(x_fit, y_fit, color=colors(i), linewidth=2)

    print(f"Cluster {i} polynomial: {poly}")

plt.title("Polynomial Clustering & Fitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
