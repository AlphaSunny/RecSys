# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:43:28 2019

@author: Pool
"""
import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)
    
    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)
    
    def GetTopN(predictions, n=10, minimumRating=4.0):
        
        
    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.copu
        
        
    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n+=1
                
        return total/n