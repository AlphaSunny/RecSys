# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:29:36 2019

@author: Pool
"""
from RecommenderMetrics import RecommenderMetrics
from EvaluatedAlgorithm import EvaluationData

class EveluatedAlgorithm:
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData, d, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if(verbose):
            print("Evaluating accuracy:")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommendorMetrics.MAE(predictions)
        
    def getName(self):
        return self.name
    
    def getAlgorithm(self):
        return self.algorithm