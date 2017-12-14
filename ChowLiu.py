"""
The implementation of Chow-Liu Tree algorithm for Categorical distribution
"""
from collections import defaultdict

import networkx as nx
import numpy as Num
import random
import copy


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
        self.error = self.error_rate()

    def marginal_distribution(self, u):
        """
        Return the marginal distribution for the u'th features of the data points, X.
        """
        values = defaultdict(float)
        for i, x in enumerate(self.X):
            values[x[u]] += self.weight[i]
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
        for i, x in enumerate(self.X):
            values[(x[u], x[v])] += self.weight[i]
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


class RandomNaiveBayes:
    def __init__(self, data, label, weight, degree, rdc):
        self.X = data
        self.label = label
        self.weight = weight
        self.lb_degree = degree
        self.rdc = rdc
        self.lb_margin = self.node_margin(self.label)
        self.lb_nb_pair_margin = {}
        self.cache = []
        self.error = self.get_lowest_error()

    def node_margin(self, node):
        values = defaultdict(float)
        for i, x in enumerate(self.X):
            values[x[node]] += self.weight[i]
        return values

    def pair_margin(self, node):
        values = defaultdict(float)
        for i, x in enumerate(self.X):
            values[(x[node], x[self.label])] += self.weight[i]
        return values

    def get_lowest_error(self):
        lowest = 1.
        for j in range(self.rdc):
            err = 0.
            temp = [1] * len(self.X)
            td = {}
            for node in random.sample(range(self.label), self.lb_degree):
                td[node] = self.pair_margin(node)
            for i, x in enumerate(self.X):
                if x[self.label] != predict_label(x, None, [self.lb_degree, self.lb_margin, td, len(self.X)]):
                    err += self.weight[i]
                    temp[i] = 0
            if err < lowest:
                lowest = err
                self.cache = copy.copy(temp)
                # no need to use deep copy
                self.lb_nb_pair_margin = copy.copy(td)
        return lowest


class RandomTree:
    def __init__(self, data, label, weight, degree):
        self.X = data
        self.label = label
        self.weight = weight
        self.lb_margin = self.node_margin(self.label)
        self.lb_nb_pair_margin = self.pair_margin(random.sample(range(label), degree))
        self.lb_degree = degree
        self.cache = [1] * len(data)
        self.error = self.error_rate()

    def node_margin(self, node):
        values = defaultdict(float)
        for i, x in enumerate(self.X):
            values[x[node]] += self.weight[i]
        return values

    def pair_margin(self, nodes):
        margins = {}
        for node in nodes:
            margins[node] = defaultdict(float)
            for i, x in enumerate(self.X):
                margins[node][(x[node], x[self.label])] += self.weight[i]
        return margins

    def error_rate(self):
        err = 0.
        for i, x in enumerate(self.X):
            if x[self.label] != predict_label(x, self):
                err += self.weight[i]
                self.cache[i] = 0
        return err


def predict_label(x, cl=None, model=None):
    if model is None:
        model = [cl.lb_degree, cl.lb_margin, cl.lb_nb_pair_margin, len(cl.X)]
    result = {}
    for lb, pr in model[1].items():
        likelihood = (1 - model[0]) * Num.log(pr)
        for k, values in model[2].items():
            # smoothing- add-one smoothing
            ad = len(values)
            p = values[(x[k], lb)]
            likelihood += (Num.log(ad / (model[-1] + ad)) if p == 0 else Num.log(p))
        result[lb] = likelihood
    return max(result, key=result.get)


if '__main__' == __name__:
    import doctest

    doctest.testmod()
