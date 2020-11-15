#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:52:33 2020

@author: sgoutin
"""

def t1(pred, reel, labels):
    reel = np.array(reel).reshape(-1,1)
    pred = np.array(pred).reshape(-1,1)
    
    return(flat_f1_score(pred, reel, average="weighted", labels=labels))    

flatten = lambda t: [item for sublist in t for item in sublist]

agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                   s["POS"].values.tolist(),
                                                   s["Tag"].values.tolist())]

word_func = lambda x: [w for w in x["Word"].values.tolist()]

tag_func = lambda x: [w for w in x["Tag"].values.tolist()]

indices_func = lambda x: list(range(len(x)))