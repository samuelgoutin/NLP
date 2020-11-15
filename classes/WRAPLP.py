#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:07:08 2020

@author: sgoutin
"""

class WrapLabelSpreading(LabelSpreading):
    """
    In order to perform a grid search over this semi-supervised model,
    we need to provide a wrapper that masks a subset of the data before
    `fit` is called.
    """
    def __init__(self, supervision_fraction, kernel='knn', gamma=20,
                 n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3, n_jobs=None):

        self.supervision_fraction = supervision_fraction

        super().__init__(kernel=kernel, gamma=gamma,
                         n_neighbors=n_neighbors, alpha=alpha,
                         max_iter=max_iter, tol=tol, n_jobs=n_jobs)

    def fit(self, X, y):
        # mask a random subset of labels, based on self.supervision_fraction
        n_total = len(y)
        n_labeled = int(self.supervision_fraction * n_total)

        indices = np.arange(n_total)
        np.random.seed(0)
        np.random.shuffle(indices)
        unlabeled_subset = indices[n_labeled:]

        y[unlabeled_subset] = -1

        super().fit(X, y)
        return(self)