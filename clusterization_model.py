import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import logging
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.patches as mpatches


class HierarchyModel:
    def __init__(self, n_clusters, **kwargs):
        self.labels_ = []
        self.Z = []
        self.linkage_params = kwargs.pop("linkage", {})
        self.fcluster_params = kwargs.pop("fcluster", {})

    def fit(self, x, y=None):
        self.Z = linkage(x, method=self.linkage_params.get('method', 'single'),
                         metric=self.linkage_params.get('metric', 'mahalanobis'))
        # print(Z[-4:, 2])
        # c, coph_dists = cophenet(Z, pdist(clients))
        # print(c)
        params = {i: self.fcluster_params[i] for i in self.fcluster_params if i != 'max_d' and i != 'criterion'}
        self.labels_ = fcluster(self.Z, self.get_max_distance(),
                                criterion=self.fcluster_params.get('criterion', 'distance'),
                                **params)
        return self

    def predict(self, x, y=None):
        return None

    def get_max_distance(self):
        return self.fcluster_params.get('max_d', 3)


class ClusterizationModel:
    """Model for clusterization

    Parameters
    -----------
    n_clusters : integer, optional
        The dimension of the projection subspace.

    Attributes
    ----------
    n_clusters : int, number of classes

    labels : list of int
        Labels of each point
    """

    def __init__(self, n_clusters=8, model="agglomerative", **kwargs):
        self.n_clusters = n_clusters
        self.labels = []
        self.X = []
        self.model_name = model
        if model == "hierarchy":
            self.base_model = HierarchyModel(n_clusters, **kwargs)
        elif model == "KMeans":
            self.base_model = KMeans(n_clusters, **kwargs)
        elif model == "agglomerative":
            self.base_model = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters, **kwargs)
        elif model == "dbscan":
            self.base_model = DBSCAN(**kwargs)
        else:
            self.base_model = cluster.SpectralClustering(n_clusters, **kwargs)
            self.model_name = "SpectralClustering"

    def _preproc_data(self, X):
        if isinstance(X, pd.DataFrame):
            return X.as_matrix()
        return X

    def fit(self, x, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies spectral clustering to this affinity matrix.

        Parameters
        ----------
        x : The input samples, shape = [n_samples, n_features]

        Returns
        -------
        self : object
            Returns self.
        """
        self.X = x
        self.base_model.fit(self._preproc_data(x), y)
        self.labels = self.base_model.labels_
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        if n_clusters_ != self.n_clusters:
            logging.warning("Clustering model provides different cluster count than expected: %s instead of %s" % (
                n_clusters_, self.n_clusters))
        self.n_clusters = n_clusters_
        return self

    def get_mean_values(self):
        data = pd.DataFrame(self.X)
        data['label'] = pd.Series(self.labels)
        result = data.groupby('label').mean()
        result['Cluster size'] = data.groupby('label').count().iloc[:, 0]
        return result

    def get_labels(self):
        return self.labels

    def get_silhouette_score(self):
        from sklearn.metrics import silhouette_score
        return silhouette_score(self.X, self.labels,
                                metric='euclidean')

    def draw_clusters(self, method=None, title=None, axis=None, show=True, **kwargs):
        data = self._preproc_data(self.X)
        reduced_data = PCA(n_components=2).fit_transform(data)
        if axis is None:
            draw_obj = plt
        else:
            draw_obj = axis
        if title is None:
            title = self.model_name + " %s clusters total" % self.n_clusters
        if method == "areas":
            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

            # Step size of the mesh. Decrease to increase the quality of the VQ.
            parts_n = kwargs.pop("parts_n", 10)
            h_x = (x_max - x_min) / parts_n  # point in the mesh [x_min, m_max]x[y_min, y_max].
            h_y = (y_max - y_min) / parts_n  # point in the mesh [x_min, m_max]x[y_min, y_max].

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

            neighbors_classifier = KNeighborsClassifier().fit(reduced_data, self.labels)
            # Obtain labels for each point in mesh. Use last trained model.
            Z = neighbors_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            draw_obj.imshow(Z, interpolation='nearest',
                            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                            cmap=plt.cm.Paired,
                            aspect='auto', origin='lower')
            draw_obj.plot(reduced_data[:, 0], reduced_data[:, 1], 'o', markersize=9)

            if "cluster_centers_" in self.base_model.__dict__:
                # Plot the centroids as a white X
                centroids = self.base_model.cluster_centers_
                plt.scatter(centroids[:, 0], centroids[:, 1],
                            marker='x', s=169, linewidths=9,
                            color='w', zorder=10)
                # plt.xlim(x_min, x_max)
                # plt.ylim(y_min, y_max)
                # plt.xticks(())
                # plt.yticks(())
        elif method == "dendrogram":
            def fancy_dendrogram(*args, **kwargs):
                max_d = kwargs.pop('max_d', None)
                if max_d and 'color_threshold' not in kwargs:
                    kwargs['color_threshold'] = max_d
                annotate_above = kwargs.pop('annotate_above', 0)

                ddata = dendrogram(*args, **kwargs)

                if not kwargs.get('no_plot', False):
                    plt.xlabel('sample index or (cluster size)')
                    plt.ylabel('distance')
                    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                        x = 0.5 * sum(i[1:3])
                        y = d[1]
                        if y > annotate_above:
                            plt.plot(x, y, 'o', c=c)
                            plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                         textcoords='offset points',
                                         va='top', ha='center')
                    if max_d:
                        plt.axhline(y=max_d, c='k')
                return ddata

            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            fancy_dendrogram(
                self.base_model.Z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=400,  # show only the last p merged clusters
                leaf_rotation=90.,  # rotates the x axis labels
                leaf_font_size=8.,  # font size for the x axis labels
                max_d=self.base_model.get_max_distance(),
                annotate_above=3,  # useful in small plots so annotations don't overlap
                show_contracted=True,  # to get a distribution impression in truncated branches
            )
        else:
            core_samples_mask = np.zeros_like(self.labels, dtype=bool)
            if 'core_sample_indices_' in self.base_model.__dict__:
                core_samples_mask[self.base_model.core_sample_indices_] = True

            # Black removed and is used for noise instead.
            unique_labels = set(self.labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = 'k'

                class_member_mask = (self.labels == k)

                xy = reduced_data[class_member_mask & core_samples_mask]
                xy2 = reduced_data[class_member_mask & ~core_samples_mask]
                draw_obj.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                              markeredgecolor='k', markersize=14)
                draw_obj.plot(xy2[:, 0], xy2[:, 1], 'o', markerfacecolor=col,
                              markeredgecolor='k', markersize=9)

        patch = mpatches.Rectangle([0, 0], 0, 0, color="black", label="score = " + str(self.get_silhouette_score()))
        draw_obj.legend(handles=[patch])
        if axis is None:
            plt.title(title)
        else:
            axis.set_title(title)
        if show:
            plt.show()
