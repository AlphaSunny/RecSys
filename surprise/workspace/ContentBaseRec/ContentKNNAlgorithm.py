#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:13:02 2019

@author: Ryan
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from MovieLens import MovieLens
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        # 计算他们的相似矩阵根据内容特性
        # 加载特性向量
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        mes = ml.getMiseEnScene()
        
        print("Computing content-based similarity matrix...")
        
        # 计算类型差异得到2*2的差异矩阵
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRating in range(self.trainset.n_items):
            if(thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
            
                
                
        print("...done.")
        return self
                
    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy =0,0,0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x* x
            sumyy += y* y
            sumxy += x*y
            
        return sumxy/math.sqrt(sumxx * sumyy)
    
    def computeYearSimilarity(self, movie1, movie2, years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff/10.0)
        return sim
    
    def computeMiseEnSceneSimilarity(self, movie1, movie2, mes):
        mes1 = mes[movie1]
        mes2 = mes[movie2]
        if (mes1 and mes2):
            shotLengthDiff = math.fabs(mes1[0] - mes2[0])
            colorVarianceDiff = math.fabs(mes1[1] - mes2[1])
            motionDiff = math.fabs(mes1[3] - mes2[3])
            lightingDiff = math.fabs(mes1[5] - mes1[5])
            numShotDiff = math.fabs(mes1[6] - mes1[6] )
            return shotLengthDiff * colorVarianceDiff * motionDiff * lightingDiff * numShotDiff      
        else:
            return 0
        
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i, rating[0]]
            neighbors.append((genreSimilarity, rating[1]))
        
        # 提取top-K最相似的排名
        k_neighbors = heapq.nlargest(self.k, neighbors, key = lambda t:t[0])
        
        # 计算平均sim得分对于k近邻，weighted靠用户评分
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if(simScore >0):
                simTotal += simScore
                weightedSum += simScore * rating
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
