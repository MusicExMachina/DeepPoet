import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from pydiffmap import diffusion_map as dm

X=np.load('Data/4wells_traj.npy')
print(X.shape)


def DW1(x):
    return 2.0 * (np.linalg.norm(x) ** 2 - 1.0) ** 2


def DW2(x):
    return 4.0 * (np.linalg.norm(x) ** 2 - 1.0) ** 2


def DW(x):
    return DW1(x[0]) + DW1(x[1])


from matplotlib import cm

mx = 5

xe = np.linspace(-mx, mx, 100)
ye = np.linspace(-mx, mx, 100)
energyContours = np.zeros((100, 100))
for i in range(0, len(xe)):
    for j in range(0, len(ye)):
        xtmp = np.array([xe[i], ye[j]])
        energyContours[j, i] = DW(xtmp)

levels = np.arange(0, 10, 0.5)
plt.contour(xe, ye, energyContours, levels, cmap=cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], s=5, c='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()

mydmap = dm.DiffusionMap(n_evecs = 2, epsilon = .2, alpha = 0.5, k=400, metric='euclidean')
dmap = mydmap.fit_transform(X)

from pydiffmap.visualization import embedding_plot

embedding_plot(mydmap, scatter_kwargs = {'c': X[:,0], 's': 5, 'cmap': 'coolwarm'})

plt.show()

from pydiffmap.visualization import data_plot

data_plot(mydmap, scatter_kwargs = {'s': 5, 'cmap': 'coolwarm'})
plt.show()

V=DW
beta=1
target_distribution=np.zeros(len(X))
for i in range(len(X)):
    target_distribution[i]=np.exp(-beta*V(X[i]))
mytdmap = dm.DiffusionMap(alpha=1.0, n_evecs = 2, epsilon = .2, k=400)
tmdmap = mytdmap.fit_transform(X, weights=target_distribution)

embedding_plot(mytdmap, scatter_kwargs = {'c': X[:,0], 's': 5, 'cmap': 'coolwarm'})

plt.show()

V=DW
beta=10
target_distribution2=np.zeros(len(X))
for i in range(len(X)):
    target_distribution2[i]=np.exp(-beta*V(X[i]))
mytdmap2 = dm.DiffusionMap( alpha = 1.0, n_evecs = 2, epsilon = .2,  k=400)
tmdmap2 = mytdmap2.fit_transform(X, weights=target_distribution2)
embedding_plot(mytdmap2, scatter_kwargs = {'c': X[:,0], 's': 5, 'cmap': 'coolwarm'})

plt.show()

plt.scatter(X[:,0], X[:,1], c = mytdmap.q, s=5, cmap=cm.coolwarm)

clb=plt.colorbar()
clb.set_label('q')
plt.xlabel('First dominant eigenvector')
plt.ylabel('Second dominant eigenvector')
plt.title('TMDmap Embedding, beta=1')

plt.show()

import scipy.sparse.linalg as spsl
P = mytdmap.P
[evals, evecs] = spsl.eigs(P.transpose(),k=1, which='LM')

phi = np.real(evecs.ravel())
q_est = phi*mytdmap.q
q_est = q_est/sum(q_est)
q_exact = target_distribution/sum(target_distribution)
print(np.linalg.norm(q_est - q_exact,1))

plt.figure(figsize=(16,6))

ax = plt.subplot(121)
ax.scatter(X[:,0], X[:,1], c = q_est, s=5, cmap=cm.coolwarm)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('estimate of pi')

ax2 = plt.subplot(122)
ax2.scatter(X[:,0], X[:,1], c = q_exact, s=5, cmap=cm.coolwarm)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('exact pi')

plt.show()