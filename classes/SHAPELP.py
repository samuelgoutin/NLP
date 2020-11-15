#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:56:35 2020

@author: sgoutin
"""

class ShapeLabelSpreading(LabelSpreading):
    """
    Pour pouvoir mener à bien l'interpolation, 
    on met en forme les prédictions issues du modèle Label Spreading
    """
    def __init__(self, kernel='knn', gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3, n_jobs=None):
        super().__init__(kernel=kernel, 
                         gamma=gamma,
                         n_neighbors=n_neighbors, 
                         alpha=alpha,
                         max_iter=max_iter, 
                         tol=tol, 
                         n_jobs=n_jobs)

    def fit(self, ind_train, ind_test, d):
        
        x_lp_train = d.x_lp(ind_train)
        X = x_lp_train + d.x_lp(ind_test)
        y = d.y_lp(ind_train) + ([-1] * len(d.y_lp(ind_test)))
        
        super().fit(X, y)
        
        distrib = self.label_distributions_[len(x_lp_train):]
        self.predictions = [{d.idx2tag[l]:i for l,i in zip(self.classes_,word)} for word in distrib]
        self.y_reel = [d.idx2tag[w] for w in d.y_lp(ind_test)]
        
        return(self)
        
