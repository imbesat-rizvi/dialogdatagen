from annoy import AnnoyIndex
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib


class Annoy:
    r"""Approximate Nearest Neighbour wrapper around Annoy library
    at https://github.com/spotify/annoy"""

    def __init__(self, feature_dim=None, metric="angular"):
        r"""f is dimension of the features and metric can be "angular",
        "euclidean", "manhattan", "hamming", or "dot"."""

        self.feature_dim = feature_dim
        self.metric = metric

        if feature_dim:
            self.index = AnnoyIndex(f=feature_dim, metric=metric)
        else:
            self.index = None

        self.n_items = 0
        self.n_trees = 0

    def __getitem__(self, idx):
        if not isinstance(idx, slice):
            return np.array(self.index.get_item_vector(idx), dtype="float32")
        return np.array(
            [self.index.get_item_vector(i) for i in range(self.n_items)[idx]],
            dtype="float32",
        )

    @classmethod
    def load(cls, path):
        r"""Load the AnnoyIndex from the provided path. The provided
        path should be of the AnnoyIndex and not the helper object."""

        path = Path(path)
        helper_path = path.parent / (path.stem + "_helper" + path.suffix)
        helper = joblib.load(helper_path)

        obj = cls(feature_dim=helper["feature_dim"], metric=helper["metric"])
        obj.n_trees = helper["n_trees"]
        obj.index.load(str(path))
        obj.n_items = obj.index.get_n_items()

        return obj

    def add_vectors(self, vectors):
        r"""add vectors to ANN"""
        for v in tqdm(vectors):
            self.index.add_item(self.n_items, v)
            self.n_items += 1

        return self

    def build(self, n_trees=10, n_jobs=-1):
        r"""Build the Annoy Index"""
        self.n_trees = n_trees
        self.index.build(n_trees=n_trees, n_jobs=n_jobs)
        return self

    def save(self, path):
        r"""Save the AnnoyIndex in the provided path"""

        self.index.save(path)
        path = Path(path)
        helper_path = path.parent / (path.stem + "_helper" + path.suffix)
        helper = dict(
            feature_dim=self.feature_dim,
            metric=self.metric,
            n_trees=self.n_trees,
        )
        joblib.dump(helper, helper_path)

    def get_knn(
        self,
        queries,
        k=-1,
        search_k=-1,
        query_as_vector=True,
        include_distances=True,
    ):
        r"""Method to return nearest neighbours for a given query which
        can either be an iterable of ids already indexed or an iterable
        of query vectors. If the query is for already indexed ids then
        set query_as_vector as False.
        """

        if k == -1:
            k = self.n_items

        if query_as_vector:
            queries = np.atleast_2d(queries)
            fetch_knn = self.index.get_nns_by_vector
        else:
            fetch_knn = self.index.get_nns_by_item

        idcs = np.empty((len(queries), k), dtype="int32")
        distances = None
        if include_distances:
            distances = np.empty((len(queries), k))

        if include_distances:
            for i, q in enumerate(tqdm(queries)):
                idcs[i], distances[i] = fetch_knn(
                    q,
                    k,
                    search_k=search_k,
                    include_distances=include_distances,
                )
        else:
            for i, q in enumerate(tqdm(queries)):
                idcs[i] = fetch_knn(
                    q,
                    k,
                    search_k=search_k,
                    include_distances=include_distances,
                )

        return idcs, distances
