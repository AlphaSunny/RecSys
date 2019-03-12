#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:09:21 2019

@author: pool
"""

from MovieLens import MovieLens
from ContentKNNAlgorithm import ContentKNNAlgorithm
from Evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# construct an evaluator to
evaluator = Evaluator(evaluationData, rankings)

contentKNN = ContentKNNAlgorithm()
evaluator.addAlgorithm(contentKNN, "ContentKNN")

# just make random recommendations
Random = NormalPredictor()
evaluator.addAlgorithm(Random, "Random")

evaluator.evaluate(False)

evaluator.sampleTopNRecs(ml)