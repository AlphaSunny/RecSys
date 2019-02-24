from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions

def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.tiem") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]

    return movieNames

def parseInput(line):
    fields = line.split()
    return Row(movieID = int(fields[1]), rating = float(fields[2]))

if __name__ == "__main__":
    spark = SparkSession.builder.appName('PopularMovies').getOrCreate()

    #获得movieName
    movieNames = loadMovieNames()

    lines = spark.sparkContext.textFile("hdfs:///user/maria_dev/ml-100k/u.data")

    movies = lines.map(parseInput)

    moviesDataset = spark.createDataFrame(movies)

    avgRating = moviesDataset.groupBy("movieID").avg("rating")

    counts = moviesDataset.groupBy("movieID").count()

    avgAndCounts = counts.join(avgRating, "movieID")

    topTen = avgAndCounts.orderBy("avg(rating)").take(10)

    for movie in topTen:
        print(movieNames[movie[0]], movie[1], movie[2])

    spark.stop()
