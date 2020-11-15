#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 01:58:44 2020

@author: sgoutin
"""

class CrossValidation:
    """
    Réalise une validation croisée pour estimer le meilleur coefficient de mixage
    dans le cadre d'une interpolation entre deux modèles.
    """
    def __init__(self, d, fold, shuffle=True):
        self.kfold = KFold(n_splits=fold, shuffle=shuffle)
        self.d = d     
        
    def estime_mixing_coef(self, pad=0.1, **kwargs):
        
        metrics = []
        seuils = np.arange(0, 1, pad).tolist()
        for train, test in self.kfold.split(self.d.grouped):
            model = Interpolation(kwargs)
            model.fit(train, test, self.d)
            seuil = [model.score(s) for s in seuils]
            metrics.append(seuil)
        res = [np.mean(val) for val in zip_longest(*metrics, fillvalue=0)]
        
        return(res)