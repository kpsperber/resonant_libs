import numpy as np
import matplotlib.pyplot as plt

N = 100
M = 10

x = np.linspace(-1, 1, N)
ys = np.linspace(-1, 1, M)

data = np.zeros((M, N))

for i in range(0, M):
    data[i] = x ** 2 + ys[i]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

for i in range(M):
    ax.plot(
        x,
        np.full_like(x, ys[i]),
        data[i]
    )

ax.set_xlabel("x")
ax.set_ylabel("Scan wavelength")
ax.set_zlabel("Intensity")

plt.tight_layout()
plt.show()