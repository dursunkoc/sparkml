package com.aric.samples.sparkml

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object LastfmImplicitRecommendation {
  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("LastfmImplicitRecommendation")
      .enableHiveSupport()
      .master("local[*]")
      .config("spark.driver.memory", "2g")
      .getOrCreate()

    import spark.implicits._

    val rawData = spark.read.option("header", "true").option("delimiter", "\t").csv("src/main/resources/data/lastfm/user_artists.dat")
      .select($"weight".cast("float"), $"userID".cast("int"), $"artistID".cast("int"))

    println("read data")
    val vectorizedDF = new VectorAssembler().setInputCols(Array("weight")).setOutputCol("weight_vec").transform(rawData)
    val std_scaler = new StandardScaler().setInputCol("weight_vec").setOutputCol("weight_std").setWithMean(true).setWithStd(true)
    println("Standardizing data")

    val vecToSeq = udf((v: Vector) => v.toArray)

    val stdDF = std_scaler.fit(vectorizedDF).transform(vectorizedDF).select($"userID", $"artistID", explode(vecToSeq($"weight_std")).as("weight_std"))

    println("Standardized data")
    println("Spliting data")
    val trainAndTestData = stdDF.randomSplit(Array(.8, .2), 21)
    val trainData = trainAndTestData(0)
    val testData = trainAndTestData(1)
    println("Splited data")

    val als = new ALS().setUserCol("userID").setItemCol("artistID").setColdStartStrategy("drop").setImplicitPrefs(true).setRatingCol("weight_std")
    println("Training")
    val alsModel = als.fit(trainData)
    println("Testing")
    val testPreds = alsModel.transform(testData)
    println("Evaluating")
    val rmseEval = new RegressionEvaluator().setMetricName("rmse").setLabelCol("weight_std").setPredictionCol("prediction")
    val rmseScore = rmseEval.evaluate(testPreds)
    println(s"The rmseScore is ${rmseScore}")

    def recommendForUser(userID: Int, numItems: Int) = {
      val dataset = List(userID).toDF("userID")
      val artistData = spark.read.option("header", "true").option("delimiter", "\t").csv("src/main/resources/data/lastfm/artists.dat")
      val recommendations = alsModel.recommendForUserSubset(dataset, numItems)
        .select(explode($"recommendations").as("recommendations"))
        .select($"recommendations.artistID".as("artistID"), $"recommendations.rating".as("rating"))
        .join(artistData, $"artistID" === $"id").orderBy($"rating").select($"id",$"rating",$"name",$"url")
      recommendations
    }
    println("Recommending")
    recommendForUser(1280, 10).show(truncate=false)
    println("Recommended")
  }
}