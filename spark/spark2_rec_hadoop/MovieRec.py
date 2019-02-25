from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import lit

def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')

    return movieNames

def parseInput(line):
    fields = line.value.split()
    return Row(userID = int(fields[0]),movieID = int(fields[1]), rating = float(fields[2]))

if __name__ == "__main__":
    spark = SparkSession.builder.appName('MovieRec').getOrCreate()

    # get movieName
    movieNames = loadMovieNames()

    lines = spark.read.text("hdfs:///user/maria_dev/ml-100k/u.data").rdd

    ratingsRdd = lines.map(parseInput)

    # convert to a dataframe and cache it
    ratings = spark.createDataFrame(ratingsRdd).cache()

    # create a als collaborative filtering model from the complete data set
    als = ALS(maxIter=5,regParam=0.01,userCol="userID", itemCol="movieID", ratingCol="rating")
    model = als.fit(ratings)

    #print out rating from user 0:
    print("\nRatings for user 0:")
    userRatings = ratings.filter("userID = 0")
    for rating in userRatings.collect():
        print(movieNames[rating['movieID']], rating['rating']) 

    print("\n top20 movie ratings")
    ratingCounts = ratings.groupBy("movieID").count().filter("count>100")

    # create two column, one is movieID, another is the userID with the value of always 0
    popularMovies = ratingCounts.select("movieID").withColumn("userID", lit(0))

    recommendations = model.transform(popularMovies)

    topRecommendations = recommendations.sort(recommendations.prediction.desc()).take(20)

    for recommendation in topRecommendations:
        print(movieNames[recommendation['movieID']], recommendation['prediction'])

    spark.stop()
    # avgRating = moviesDataset.groupBy("movieID").avg("rating")
    

    # counts = moviesDataset.groupBy("movieID").count()

    # avgAndCounts = counts.join(avgRating, "movieID")

    # topTen = avgAndCounts.orderBy("avg(rating)").take(10)

    # for movie in topTen:
    #     print(movieNames[movie[0]], movie[1], movie[2])

    # spark.stop()
