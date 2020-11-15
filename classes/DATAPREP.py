#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:17:55 2020

@author: sgoutin
"""

class Dataprep(object): # n le nombre de phrases pour le jeu de données
    """
    Préparation des données qui alimentent 
    les modèles (CRF et LabelSpreading).
    """
    def __init__(self, data):

        self.data = data
        self.grouped = data.groupby(by="Sentence #", sort=False)
        self.tags = list(set(data["Tag"].values))
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}
        
        words = data.Word.tolist()
        self.word_counter = {x: words.count(x) for x in words}
        self.no_of_sentences = len(self.grouped)
        self.indices = list(range(self.no_of_sentences))
        
    def add_w2v(self, model):
        self.w2v = model

    def x_lp(self, indices = indices_func):
        lp_x = self.grouped.apply(word_func)[indices]
        j = Dataprep.word2ngram(window=2, sentences=lp_x)
        return([self.ngram2vec(w) for w in j])
    
    def y_lp(self, indices = indices_func):
        lp_y = self.grouped.apply(tag_func)[indices]
        return([self.tag2idx[w] for s in lp_y for w in s])

    def x_crf(self, indices = indices_func):
        sent = self.grouped.apply(agg_func)[indices]
        return([Dataprep.sent2features(s) for s in sent])
    
    def y_crf(self, indices = indices_func):
        sent = self.grouped.apply(agg_func)[indices]
        return([Dataprep.sent2labels(s) for s in sent])
    
    def ngram2vec(self, list_of_ngram):
        
        no_of_sentences, word_counter = self.no_of_sentences, self.word_counter
        sent_emd = []
        for word in list_of_ngram:
            tf = word_counter[word]/float(len(list_of_ngram))
            idf = np.log(no_of_sentences/float(1+word_counter[word]))
            try:
                emd = tf*idf*self.w2v[word]
                sent_emd.append(emd)
            except:
                continue
        sent_emd = np.array(sent_emd)
        sum_ = sent_emd.sum(axis=0)
        return(sum_/np.sqrt((sum_**2).sum()))
    

    
    @staticmethod    
    def sent2features(sent):
        return [Dataprep.word2features(sent, i) for i in range(len(sent))]
    @staticmethod 
    def sent2labels(sent):
        return [label for token, postag, label in sent]
    @staticmethod 
    def sent2tokens(sent):
        return [token for token, postag, label in sent]
    
    @staticmethod
    def take_window(sent, idc, window):
        l = [x-2 for x in range(idc,idc+2*window+1) if ((x-2)>=0 and (x-2)<len(sent))]
        return([sent[w] for w in l])
    
    @staticmethod
    def word2ngram(window, sentences):
        w = [Dataprep.take_window(sent, idc, window) for sent in sentences for idc in range(len(sent))]
        return(w)   
    
    @staticmethod
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
            }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
                })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
                })
        else:
            features['EOS'] = True

        return features