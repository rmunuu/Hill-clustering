import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import (make_circles, make_moons, make_blobs)
from mpl_toolkits.mplot3d import Axes3D

# Cricles
X1 = make_circles(factor=0.5, noise=0.05, n_samples=500)

# # Moons
X2 = make_moons(n_samples=500, noise=0.05)

X3 = make_blobs(n_samples=500, random_state=42, cluster_std=3)

X = X3[0]

s = 1
# Function to generate a Gaussian peak at each data point
def gaussian(x, y, x0, y0, sigma=s):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# Meshgrid for evaluation
grid = 400
x = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, grid+1)
y = np.linspace(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1, grid+1)
x, y = np.meshgrid(x, y)

# Compute the sum of Gaussians at each grid point
z = np.zeros_like(x)
for point in X:
    z += gaussian(x, y, point[0], point[1])

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_title('3D Gaussian Plot over Dataset Points')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Intensity')


# add ball
ball = 100
ball_x = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, ball+1)
ball_y = np.linspace(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1, ball+1)
ball_x, ball_y = np.meshgrid(ball_x, ball_y)

ball_z = np.array([[z[i][j] for j in np.arange(0, 400+400//ball, 400//ball)] for i in np.arange(0, 400+400//ball, 400//ball)])

ax.scatter3D(ball_x, ball_y, ball_z, c='r', s=3)



plt.show()