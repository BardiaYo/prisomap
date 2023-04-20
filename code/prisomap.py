

"""Parzan window-Isomap for manifold learning"""


# This code is adopted from Jake Vanderplas and modified to implement parzan window constraint isometric mapping. 
# Bardia Yousefi (bardia.yousefi@gmail.com)
#  Modified 2021 under that same initial license 
# License: BSD 3 clause (C) 2011




# Original author and credit to: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_isomap.py
# License: BSD 3 clause (C) 2011

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer


class Isomap(TransformerMixin, BaseEstimator):
    """Isomap Embedding

    Non-linear dimensionality reduction through Isometric Mapping

    Read more in the :ref:`User Guide <isomap>`.

    Parameters
    ----------
    n_neighbors : integer
        number of neighbors to consider for each point.

    n_components : integer
        number of coordinates for the manifold

    eigen_solver : ['auto'|'arpack'|'dense']
        'auto' : Attempt to choose the most efficient solver
        for the given problem.

        'arpack' : Use Arnoldi decomposition to find the eigenvalues
        and eigenvectors.

        'dense' : Use a direct solver (i.e. LAPACK)
        for the eigenvalue decomposition.

    tol : float
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.

    max_iter : integer
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.

    path_method : string ['auto'|'FW'|'D']
        Method to use in finding shortest path.

        'auto' : attempt to choose the best algorithm automatically.

        'FW' : Floyd-Warshall algorithm.

        'D' : Dijkstra's algorithm.

    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    metric : string, or callable, default="minkowski"
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`.

        .. versionadded:: 0.22

    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        .. versionadded:: 0.22

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.22

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    kernel_pca_ : object
        :class:`~sklearn.decomposition.KernelPCA` object used to implement the
        embedding.

    nbrs_ : sklearn.neighbors.NearestNeighbors instance
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.

    dist_matrix_ : array-like, shape (n_samples, n_samples)
        Stores the geodesic distance matrix of training data.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import Isomap
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = Isomap(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)

    References
    ----------

    .. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
           framework for nonlinear dimensionality reduction. Science 290 (5500)
    """

    def __init__(self, n_neighbors=129, n_components=2, eigen_solver='auto', # X.shape[1]-1
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', n_jobs=None, metric='minkowski',
                 p=2, metric_params=None, radius=1.0, rdx = 0): 
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.radius = radius # Additive h from the parzen window size.
        self.rdx = rdx # selecting the mode of the isometric mapping, whether 0 standard isomap and 1 PR-isomap.

    def _fit_transform(self, X):

        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      metric=self.metric, p=self.p,
                                      metric_params=self.metric_params,
                                      n_jobs=self.n_jobs, radius = self.radius)
        self.nbrs_.fit(X)

        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)
        if self.rdx == 0:

          kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                               metric=self.metric, p=self.p,
                               metric_params=self.metric_params,
                               mode='distance', n_jobs=self.n_jobs) # radius = self.radius
                              #   See _graphy.py code indicating all of these points updated
        else:

          kng = radius_neighbors_graph(self.nbrs_, radius = self.radius, mode='connectivity',
                                metric='minkowski', p=self.p, metric_params=self.metric_params,
                                include_self=False, n_jobs=self.n_jobs)

        self.dist_matrix_ = graph_shortest_path(kng,
                                                method=self.path_method,
                                                directed=False)
        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)

    @deprecated("Attribute `training_data_` was deprecated in version 0.22 and"
                " will be removed in 0.24.")
    @property
    def training_data_(self):
        check_is_fitted(self)
        return self.nbrs_._fit_X

    def reconstruction_error(self):
        """Compute the reconstruction error for the embedding.

        Returns
        -------
        reconstruction_error : float

        Notes
        -----
        The cost function of an isomap embedding is

        ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

        Where D is the matrix of distances for the input data X,
        D_fit is the matrix of distances for the output embedding X_fit,
        and K is the isomap kernel:

        ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
        """
        G = -0.5 * self.dist_matrix_ ** 2
        G_center = KernelCenterer().fit_transform(G)
        evals = self.kernel_pca_.lambdas_
        return np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]


    # def radius_neighbors_graph(self, X):#X, radius, *, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None):
    #     """Computes the (weighted) graph of Neighbors for points in X

    #     Neighborhoods are restricted the points at a distance lower than
    #     radius.

    #     Read more in the :ref:`User Guide <unsupervised_neighbors>`.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features) or BallTree
    #         Sample data, in the form of a numpy array or a precomputed
    #         :class:`BallTree`.

    #     radius : float
    #         Radius of neighborhoods.

    #     mode : {'connectivity', 'distance'}, default='connectivity'
    #         Type of returned matrix: 'connectivity' will return the connectivity
    #         matrix with ones and zeros, and 'distance' will return the distances
    #         between neighbors according to the given metric.

    #     metric : str, default='minkowski'
    #         The distance metric used to calculate the neighbors within a
    #         given radius for each sample point. The DistanceMetric class
    #         gives a list of available metrics. The default distance is
    #         'euclidean' ('minkowski' metric with the param equal to 2.)

    #     p : int, default=2
    #         Power parameter for the Minkowski metric. When p = 1, this is
    #         equivalent to using manhattan_distance (l1), and euclidean_distance
    #         (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    #     metric_params : dict, default=None
    #         additional keyword arguments for the metric function.

    #     include_self : bool or 'auto', default=False
    #         Whether or not to mark each sample as the first nearest neighbor to
    #         itself. If 'auto', then True is used for mode='connectivity' and False
    #         for mode='distance'.

    #     n_jobs : int, default=None
    #         The number of parallel jobs to run for neighbors search.
    #         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    #         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    #         for more details.

    #     Returns
    #     -------
    #     A : sparse matrix of shape (n_samples, n_samples)
    #         Graph where A[i, j] is assigned the weight of edge that connects
    #         i to j. The matrix is of CSR format.

    #     Examples
    #     --------
    #     >>> X = [[0], [3], [1]]
    #     >>> from sklearn.neighbors import radius_neighbors_graph
    #     >>> A = radius_neighbors_graph(X, 1.5, mode='connectivity',
    #     ...                            include_self=True)
    #     >>> A.toarray()
    #     array([[1., 0., 1.],
    #           [0., 1., 0.],
    #           [1., 0., 1.]])

    #     See Also
    #     --------
    #     kneighbors_graph

    #     self.n_neighbors = n_neighbors
    #     self.n_components = n_components
    #     self.eigen_solver = eigen_solver
    #     self.tol = tol
    #     self.max_iter = max_iter
    #     self.path_method = path_method
    #     self.neighbors_algorithm = neighbors_algorithm
    #     self.n_jobs = n_jobs
    #     self.metric = metric
    #     self.p = p
    #     self.metric_params = metric_params
    #     self.radius = radius



    #     """
    #     if not isinstance(X, RadiusNeighborsMixin):
    #         X = NearestNeighbors(radius = self.radius, metric = self.metric, p=self.p,
    #                             metric_params=self.metric_params, n_jobs=self.n_jobs).fit(X)
    #     else:
    #         _check_params(X, self.metric, self.p, self.metric_params)

    #     query = _query_include_self(X._fit_X, include_self, mode)
    #     return X.radius_neighbors_graph(query, self.radius, mode)


    # def _check_params(X, metric, p, metric_params):
    #     """Check the validity of the input parameters"""
    #     params = zip(['metric', 'p', 'metric_params'],
    #                 [metric, p, metric_params])
    #     est_params = X.get_params()
    #     for param_name, func_param in params:
    #         if func_param != est_params[param_name]:
    #             raise ValueError(
    #                 "Got %s for %s, while the estimator has %s for "
    #                 "the same parameter." % (
    #                     func_param, param_name, est_params[param_name]))


    # def _query_include_self(X, include_self, mode):
    #     """Return the query based on include_self param"""
    #     if include_self == 'auto':
    #         include_self = mode == 'connectivity'

    #     # it does not include each sample as its own neighbors
    #     if not include_self:
    #         X = None

    #     return X





    def fit(self, X, y=None):
        """Compute the embedding vectors for data X

        Parameters
        ----------
        X : {array-like, sparse graph, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, sparse graph, precomputed tree, or NearestNeighbors
            object.

        y : Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse graph, BallTree, KDTree}
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features)
            If neighbors_algorithm='precomputed', X is assumed to be a
            distance matrix or a sparse graph of shape
            (n_queries, n_samples_fit).

        Returns
        -------
        X_new : array-like, shape (n_queries, n_components)
        """
        check_is_fitted(self)
        distances, indices = self.nbrs_.kneighbors(X, return_distance=True)

        # Create the graph of shortest distances from X to
        # training data via the nearest neighbors of X.
        # This can be done as a single array operation, but it potentially
        # takes a lot of memory.  To avoid that, use a loop:

        n_samples_fit = self.nbrs_.n_samples_fit_
        n_queries = distances.shape[0]
        G_X = np.zeros((n_queries, n_samples_fit))
        for i in range(n_queries):
            G_X[i] = np.min(self.dist_matrix_[indices[i]] +
                            distances[i][:, None], 0)

        G_X **= 2
        G_X *= -0.5

        return self.kernel_pca_.transform(G_X)
