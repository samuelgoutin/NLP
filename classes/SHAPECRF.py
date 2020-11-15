#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 23:23:55 2020

@author: sgoutin
"""

class ShapeCrf(CRF):
    """
    Pour pouvoir mener à bien l'interpolation, 
    on met en forme les prédictions issues du modèle CRF
    """
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False):

        super().__init__(algorithm=algorithm, 
                         c1=c1, 
                         c2=c2, 
                         max_iterations=max_iterations, 
                         all_possible_transitions=all_possible_transitions)

    def fit(self, ind_train, ind_test, d):
        
        X, y = d.x_crf(ind_train), d.y_crf(ind_train) # train
        x_test, y_test = d.x_crf(ind_test), d.y_crf(ind_test) # test
        
        super().fit(X, y)
        
        self.predictions = flatten(super().predict_marginals(x_test))
        self.y_reel = flatten(y_test)

        return(self)
    
