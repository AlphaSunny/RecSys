# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:21:17 2019

@author: Ryan
"""

from MovieLens import MovieLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# 加载数据
(ml, evaluationData, ranking) = LoadMovieLensData()

# 
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

# SVD++
SVDpp = SVDpp()
evaluator.AddAlgorithm(SVDpp, "SVD++")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)

