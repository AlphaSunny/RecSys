# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:34:12 2019

@author: Ryan
概要： 根据用户的协同过滤，来推荐电影
    1.获得评分矩阵
    2.获得相似用户矩阵
    3.候选电影产生
    4.候选电影排序
    5.候选电影过滤
特点： 无法预测评分，仅仅推荐电影
* 无法
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

testSubject = '85'
k = 10

# 加载数据
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

# 在这里可以使用不同的相似度方法
sim_options = {'name': 'cosine',
               'user_based':True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# 得到top N相似的用户, 也可以设置一个相似的阙值
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if(innerID != testUserInnerID):
        similarUsers.append((innerID, score))

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

# 得到他们评分的项，然后根据相似度来做权重
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1]/ 5.0) * userSimilarityScore;
        
# 需要排除已经看过的电影
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# 得到最高的评分电影对于相似的用户
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getMovieName(int(movieID)), ratingSum)
        pos +=1
        if(pos>10):
            break
    