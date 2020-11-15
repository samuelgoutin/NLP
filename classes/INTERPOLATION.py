#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:39:20 2020

@author: sgoutin
"""

class Interpolation(object):
    """
    Contruit un nouveau modèle en interpolant un modèle LabelSpreading et CRF.
    """
    def __init__(self, param):
        
        self.lp = ShapeLabelSpreading(kernel= param.get("kernel", "knn"), 
                                      alpha=param.get("alpha", 0.2), 
                                      n_neighbors=param.get("n_neighbors", 9), 
                                      max_iter=param.get("max_iter", 30))
        
        self.crf = ShapeCrf(algorithm= param.get("algorithm", 'lbfgs'), 
                            c1= param.get("c1", 0.1), 
                            c2= param.get("c2", 0.2), 
                            max_iterations= param.get("max_iterations", 100), 
                            all_possible_transitions= param.get("all_possible_transitions", False))
    
    def fit(self, ind_train, ind_test, d):
        self.d = d
        self.lp.fit(ind_train, ind_test, d)
        self.crf.fit(ind_train, ind_test, d)
        return(self)
        
    def predict(self, seuil):
        
        a = self.lp.predictions
        b = self.crf.predictions 

        n=[]
        for i,j in zip(a, b):  
            m={}
            for k in set(i) & set(j):
                m[k] = (i[k]*seuil + j[k]*(1-seuil))
            n.append(m)
        
        return([max(w.items(), key=operator.itemgetter(1))[0] for w in n])        
    
    def score(self, seuil):
        w = self.predict(seuil)
        t1_score = t1(w, self.crf.y_reel, [w for w in self.d.tags if w!="O"])
        return(t1_score)