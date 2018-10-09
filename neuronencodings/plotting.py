import numpy as np
from matplotlib import pyplot as plt


def plot_2d(data, plot_axis=[[0, 1]], labels=None):
    cm = plt.get_cmap('tab10')
    colors = [cm(.05 + .1 * i) for i in range(10)]

    if labels is None:
        labels = np.ones(len(data))

    u_labels = np.unique(labels)
    if len(u_labels) == 1:
        colors = [".3"]

    for axis in plot_axis:
        fig = plt.figure(figsize=(10, 10))
        fig.patch.set_facecolor('white')

        for i_label, u_label in enumerate(u_labels):
            m = labels == u_label
            plt.scatter(data[m, axis[0]], data[m, axis[1]], alpha=.7,
                        c=colors[i_label])

        plt.xlabel("Dim %d" % (axis[0] + 1), fontsize=16)
        plt.ylabel("Dim %d" % (axis[1] + 1), fontsize=16)

        plt.tight_layout()
        plt.show()