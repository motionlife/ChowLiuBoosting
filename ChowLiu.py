"""
The implementation of Chow-Liu Tree algorithm for Categorical distribution
"""
from collections import defaultdict

import networkx as nx
import numpy as Num


class ChowLiuTree:
    def __init__(self, data, label, weight):
        self.X = data
        self.label = label
        self.weight = weight
        self.lb_margin = {}
        self.lb_nb_pair_margin = {}
        self.tree = self.build_chow_liu_tree(len(self.X[0]))
        self.lb_degree = self.tree.degree(label)
        self.extract_neighbors()
        self.cache = [1] * len(data)

    def marginal_distribution(self, u):
        """
        Return the marginal distribution for the u'th features of the data points, X.
        """
        values = defaultdict(float)
        s = 1. / len(self.X)
        for i, x in enumerate(self.X):
            values[x[u]] += s * self.weight[i]
        if u == self.label:
            self.lb_margin = values
        return values

    def marginal_pair_distribution(self, u, v):
        """
        Return the marginal distribution for the u'th and v'th features of the data points, X.
        """
        if u > v:
            u, v = v, u
        values = defaultdict(float)
        s = 1. / len(self.X)
        for i, x in enumerate(self.X):
            values[(x[u], x[v])] += s * self.weight[i]
        if v == self.label:
            self.lb_nb_pair_margin[u] = values
        return values

    def calculate_mutual_information(self, u, v):
        """
        X are the data points.
        u and v are the indices of the features to calculate the mutual information for.
        """
        if u > v:
            u, v = v, u
        marginal_u = self.marginal_distribution(u)
        marginal_v = self.marginal_distribution(v)
        marginal_uv = self.marginal_pair_distribution(u, v)
        info = 0.
        for x_u, p_x_u in marginal_u.items():
            for x_v, p_x_v in marginal_v.items():
                if (x_u, x_v) in marginal_uv:
                    p_x_uv = marginal_uv[(x_u, x_v)]
                    info += p_x_uv * (Num.log(p_x_uv) - Num.log(p_x_u) - Num.log(p_x_v))
        return info

    def build_chow_liu_tree(self, n):
        """
        Build a Chow-Liu tree from the data, X. n is the number of features. The weight on each edge is
        the negative of the mutual information between those features. The tree is returned as a networkx
        object.
        """
        G = nx.Graph()
        for v in range(n):
            G.add_node(v)
            for u in range(v):
                G.add_edge(u, v, weight=-self.calculate_mutual_information(u, v))
        tree = nx.minimum_spanning_tree(G)  # (G, weight='weight', algorithm='kruskal',ignore_nan=False)
        return tree

    def extract_neighbors(self):
        """
        Return the useful information from the tree that could be used as a classifier for lth feature.
        (the label) i.e. all the neighbours of label node.
        """
        neighbors = self.tree.neighbors(self.label)
        self.lb_nb_pair_margin = {k: self.lb_nb_pair_margin[k] for k in neighbors}

    def error_rate(self):
        err = 0.
        for i, x in enumerate(self.X):
            if x[self.label] != predict_label(x, self):
                err += self.weight[i]
                self.cache[i] = 0
        return err


def predict_label(vector, cl=None, pack=None):
    if pack is None:
        pack = [cl.lb_degree, cl.lb_margin, cl.lb_nb_pair_margin]
    values = defaultdict(float)
    for lb, prob in pack[1].items():
        likely = 1 / (prob ** (pack[0] - 1))
        for nb, dist in pack[2].items():
            likely *= dist[vector[nb], lb]
        values[lb] = likely
    return max(values, key=values.get)


if '__main__' == __name__:
    import doctest

    doctest.testmod()
