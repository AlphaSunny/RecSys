# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:38:10 2019

@author: Pool
"""

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    def __init__(self, data, popularityRankings):
        self.rankings = popularityRankings
        
        # Build a full training set for evaluating overall properties
        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()
        
        # bulid  a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size = 0.25, random_state = 1)
        
        #Build a "leave one out" train/test split for evaluating top-N recommenders
        #And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test
        
        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()
        
        # Compute similarity matrix between items so we can measure diversity
        sim_options = {'name':'cosine', 'user_based':False}
        self.simsAlgo = KNNBaseline(sim_options = sim_options)
        self.simsAlgo.fit(self.fullTrainSet)
        
    def getFullTrainSet(self):
        return self.fullTrainSet
    
    def getFullAntiTestSet(self):
        return self.fullAntiTestSet
    
    def getAntiTestSetForUser(self, testSubject):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset
        
        
        
    def getTrainSet(self):
        return self.trainSet
    
    def getTestSet(self):
        return self.testSet
    
    def getLOOCVTrainSet(self):
        return self.LOOCVTrain
    
    def getLOOCVTestSet(self):
        return self.LOOCVTest

    def getLOOCVAntiTestSet(self):
        return self.LOOCVAntiTestSet
    
    def getSimilarities(self):
        return self.simsAlgo
    
    def getPopularityRankings(self):
        return self.rankings