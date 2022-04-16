import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_embedded_vecs(vecs):
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vecs)
    plt.scatter(vecs_2d[:,0], vecs_2d[:,1])
    for i, point in enumerate(vecs_2d):
        plt.annotate(i, (point[0], point[1]))
    plt.show()



def plot_HRK(hrk_list, titles_list):
    y_axis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    for HRK, title in zip(hrk_list, titles_list):
        plt.plot(HRK, label=title)

    plt.ylim([0, 1])
    plt.yticks(y_axis)
    plt.xlim([1, 11])
    plt.legend()
    plt.grid()
    plt.show()
