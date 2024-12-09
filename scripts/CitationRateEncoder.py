from get_graph import get_digraph
import networkx as nx
import numpy as np
from sklearn.exceptions import NotFittedError


class CitationRateEncoder:
    def __init__(self, n_classes: int):
        """
        Encoder of citation rate (in-degree) of a graph into classes. Class
        boundaries are defined in a way that produces the most balanced
        classes. This encoder follows `sklearn` interface and implements
        methods `fit`, `transform` and `inverse_transform`.

        Attributes
        ----------
        classes_ : ndarray of shape (n_classes, 2)
            Holds the boundaries for each class.

        Parameters
        ----------
        n_classes : int
            Number of classes to split the in degrees into.

        Examples
        --------
        Fitting and encoding some in degrees can be done this way:
        >>> from CitationRateEncoder import CitationRateEncoder
        >>> encoder = CitationRateEncoder(5).fit()
        >>> encoder.transform([15, 1000])
        array([3, 4])
        >>> encoder.inverse_transform([3, 4])
        array([[   7,   15],
               [  16, 2414]])

        Notes
        -----
        Resulting classes are not perfectrly even due to only integer values
        possible. Class 0 will usually be the biggest one since the graph
        follows power law and has significantly more nodes with small in degree
        than ones with big in degree.
        """
        self.n_classes = n_classes

    def fit(self, G: nx.DiGraph | None = None):
        """
        Define class boundaries based on the graph and requested number of
        classes.

        Parameters
        ----------
        G: nx.DiGraph, optional
            Graph to analyze. If not provided, `get_digraph` function is called
            with default parameters to get the graph from intended project
            structure (see `get_digraph` for more info).

        Returns
        -------
        self: object
            Fitted encoder instance.
        """
        if G is None:
            G: nx.DiGraph = get_digraph()
        classes = []
        bins = []
        all_nodes = np.array(list(dict(G.in_degree).values()))
        for i in range(self.n_classes):
            left = int(np.quantile(all_nodes, i / self.n_classes)) + (i != 0)
            right = int(np.quantile(all_nodes, (i + 1) / self.n_classes))
            classes.append(np.array([left, right]))
            bins.append(left)
        self.bins_ = np.array(bins)
        self.classes_ = np.array(classes)
        return self

    def transform(self, y: int | np.ndarray) -> int | np.ndarray:
        """
        Encode the in-degrees into classes.

        Parameters
        ----------
        y : int or ndarray of int
            A single value or values to encode.

        Returns
        -------
        classified : int or ndarray of int
            Encoded value or values.
        """
        if hasattr(self, 'bins_'):
            classified = np.digitize(y, self.bins_, right=False) - 1
            return classified
        else:
            raise NotFittedError("Encoder has not been fitted.")

    def inverse_transform(self, y: int | np.ndarray) -> int | np.ndarray:
        """
        Decode classes into their possible values.

        Parameters
        ----------
        y : int or ndarray of int
            A single label or labels to decode.

        Returns
        -------
        decoded : ndarray oof shape (`y.shape[0]`, 2)
            Class boundaries for each label in `y`.
        """
        if hasattr(self, 'bins_'):
            return self.classes_[y]
        else:
            raise NotFittedError("Encoder has not been fitted.")


if __name__ == '__main__':
    encoder = CitationRateEncoder(10).fit()
    classified = encoder.transform(15)
    print(classified)
    print(encoder.inverse_transform(classified))
    print(encoder.classes_)
