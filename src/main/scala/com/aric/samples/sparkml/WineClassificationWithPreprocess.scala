package com.aric.samples.sparkml

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SaveMode

object WineClassificationWithPreprocess extends App {

  val spark = SparkSession.builder().appName("WineClassificationWithPreprocess_APP").master("local[*]").getOrCreate()

  import spark.sqlContext.implicits._

  val data = spark.read.option("header", false).csv("src/main/resources/data/wine.data")

  val df = data.toDF(
    "Label",
    "Alcohol",
    "MalicAcid",
    "Ash",
    "AshAlkalinity",
    "Magnesium",
    "TotalPhenols",
    "Flavanoids",
    "NonflavanoidPhenols",
    "Proanthocyanins",
    "ColorIntensity",
    "Hue",
    "OD",
    "Proline").select(
      $"Label".cast("Double"),
      $"Alcohol".cast("Double"),
      $"MalicAcid".cast("Double"),
      $"Ash".cast("Double"),
      $"AshAlkalinity".cast("Double"),
      $"Magnesium".cast("Double"),
      $"TotalPhenols".cast("Double"),
      $"Flavanoids".cast("Double"),
      $"NonflavanoidPhenols".cast("Double"),
      $"Proanthocyanins".cast("Double"),
      $"ColorIntensity".cast("Double"),
      $"Hue".cast("Double"),
      $"OD".cast("Double"),
      $"Proline".cast("Double"))

  val vdf = new VectorAssembler().setInputCols(Array(
    "Alcohol",
    "MalicAcid",
    "Ash",
    "AshAlkalinity",
    "Magnesium",
    "TotalPhenols",
    "Flavanoids",
    "NonflavanoidPhenols",
    "Proanthocyanins",
    "ColorIntensity",
    "Hue",
    "OD",
    "Proline")).setOutputCol("features").transform(df).drop(
    "Alcohol",
    "MalicAcid",
    "Ash",
    "AshAlkalinity",
    "Magnesium",
    "TotalPhenols",
    "Flavanoids",
    "NonflavanoidPhenols",
    "Proanthocyanins",
    "ColorIntensity",
    "Hue",
    "OD",
    "Proline")

  val vdfIndexed = new StringIndexer().setInputCol("Label").setOutputCol("Label_index").fit(vdf).transform(vdf)

  val testAndTrainingData = vdfIndexed.randomSplit(Array(.8, .2), 42)

  val trainingData = testAndTrainingData(0)
  val testData = testAndTrainingData(1)

  println(trainingData.show(10))
  println(testData.show(10))

  val dtc = new DecisionTreeClassifier()
    .setLabelCol("Label_index")
    .setFeaturesCol("features")
    .setMaxDepth(3)
    .setMaxBins(32)
    .setImpurity("gini")
    .setPredictionCol("prediction")

  val model: DecisionTreeClassificationModel = dtc.fit(trainingData)

  print(model)

  val predictions = model.transform(testData)

  List("f1", "accuracy").foreach(metric => {
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Label_index").setPredictionCol("prediction").setMetricName(metric)
    val result = evaluator.evaluate(predictions)
    println(s"${evaluator.getMetricName} is ${result}")
  })

}