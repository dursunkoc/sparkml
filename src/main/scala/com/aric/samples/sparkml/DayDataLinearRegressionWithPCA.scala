package com.aric.samples.sparkml

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object DayDataLinearRegressionWithPCA extends App {
  val spark = SparkSession.builder().appName("DayDataLinearRegressionWithPCA").master("local[*]").enableHiveSupport().getOrCreate()

  import spark.implicits._

  val rawData = spark.read.option("header", "true").csv("src/main/resources/data/day.csv")

  val dataFrame = rawData.select($"season".cast("float"),
    $"yr".cast("float"),
    $"mnth".cast("float"),
    $"holiday".cast("float"),
    $"weekday".cast("float"),
    $"workingday".cast("float"),
    $"weathersit".cast("float"),
    $"temp".cast("float"),
    $"atemp".cast("float"),
    $"hum".cast("float"),
    $"windspeed".cast("float"),
    $"cnt".cast("float"))

  val featureCols = dataFrame.columns.filterNot(_=="cnt")

  val vecDF = new VectorAssembler().setInputCols(featureCols).setOutputCol("feature").transform(dataFrame)

  val trainAndTestData = vecDF.randomSplit(Array(.8,.2), 21)
  val trainData = trainAndTestData(0)
  val testData = trainAndTestData(1)

  val lr = new LinearRegression().setFeaturesCol("feature").setLabelCol("cnt").setPredictionCol("prediction").setElasticNetParam(.8).setMaxIter(50).setRegParam(1.0)
  val lrModel = lr.fit(trainData)

  val lrTrainPreds = lrModel.transform(trainData)
  val lrTestPreds = lrModel.transform(testData)


  val pca = new PCA().setK(8).setInputCol("feature").setOutputCol("pcaFeature")
  val pcaModel = pca.fit(trainData)

  val pcaTrainData = pcaModel.transform(trainData)
  val pcaTestData = pcaModel.transform(testData)

  val lrPca = new LinearRegression().setFeaturesCol("pcaFeature").setLabelCol("cnt").setPredictionCol("pcaPrediction").setElasticNetParam(.8).setMaxIter(50).setRegParam(1.0)
  val lrPcaModel = lrPca.fit(pcaTrainData)

  val lrPcaTrainPreds = lrPcaModel.transform(pcaTrainData)
  val lrPcaTestPreds = lrPcaModel.transform(pcaTestData)


  val r2Eval = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("cnt").setMetricName("r2")
  val rmseEval = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("cnt").setMetricName("rmse")
  val r2PcaEval = new RegressionEvaluator().setPredictionCol("pcaPrediction").setLabelCol("cnt").setMetricName("r2")
  val rmsePcaEval = new RegressionEvaluator().setPredictionCol("pcaPrediction").setLabelCol("cnt").setMetricName("rmse")

  println(s"The r2 value for Train is ${r2Eval.evaluate(lrTrainPreds)}")
  println(s"The RMSE value for Train is ${rmseEval.evaluate(lrTrainPreds)}")
  println(s"The r2 value for Test is ${r2Eval.evaluate(lrTestPreds)}")
  println(s"The RMSE value for Test is ${rmseEval.evaluate(lrTestPreds)}")
  println(s"The r2 value for PCA Train is ${r2PcaEval.evaluate(lrPcaTrainPreds)}")
  println(s"The RMSE value for PCA Train is ${rmsePcaEval.evaluate(lrPcaTrainPreds)}")
  println(s"The r2 value for PCA Test is ${r2PcaEval.evaluate(lrPcaTestPreds)}")
  println(s"The RMSE value for PCA Test is ${rmsePcaEval.evaluate(lrPcaTestPreds)}")

}
