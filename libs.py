#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 02:01:10 2020

@author: sgoutin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from gensim.models import Word2Vec
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import cross_val_predict, cross_validate, cross_val_score
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from sklearn.metrics import precision_score, f1_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_crfsuite import CRF
import operator        
from itertools import zip_longest
