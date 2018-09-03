package com.aric.samples.sparkml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types.StructType

object MovilensRecommendations {

  def main(arg: Array[String]) {
    val spark = SparkSession.builder().appName("MovilensRecommendations").master("local[*]").config("spark.driver.memory", "2G").getOrCreate()
    import spark.implicits._
    val rawData = spark.read.option("header", "true").csv("src/main/resources/data/movielens/ratings.csv")
    val dataFrame = rawData.select($"userId".cast("int"), $"movieId".cast("int"), $"rating".cast("float"))
    dataFrame.show()

    val trainTestData = dataFrame.randomSplit(Array(0.8, .02), 21)
    val trainData = trainTestData(0)
    val testData = trainTestData(1)

    val als = new ALS().setMaxIter(10)
    .setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
    .setColdStartStrategy("drop").setRegParam(.1)
    val alsModel = als.fit(trainData)
    val testPreds = alsModel.transform(testData)
    testPreds.describe("rating", "prediction").show(truncate = false)

    val regEval = new RegressionEvaluator().setLabelCol("rating").setPredictionCol("prediction").setMetricName("rmse")
    val rmseResult = regEval.evaluate(testPreds)
    println(s"The RMSE result is ${rmseResult}")
    import org.apache.spark.sql.functions._

    def recommendationsForUser(userId: Int, maxItems: Int) = {
      val subSet = List(userId).toDF("userId")
      val recommendations = alsModel.recommendForUserSubset(subSet, maxItems)
        .select(explode($"recommendations").as("recommendation"))
        .select($"recommendation.movieId".as("movieId"), $"recommendation.rating".as("rating"))
        
     val movies = spark.read.option("header", "true").csv("src/main/resources/data/movielens/movies.csv")
     recommendations.join(movies, Seq("movieId"), "inner").orderBy("rating").select("title", "genres", "rating")
    }
    val extracted = recommendationsForUser(3, 10)
    extracted.show(truncate= false)

  }
}