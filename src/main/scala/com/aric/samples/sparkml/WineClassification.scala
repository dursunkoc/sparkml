package com.aric.samples.sparkml

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils

object WineClassification {

  def main(args: Array[String]): Unit = {
    nonScaledClassificatio()
    scaledClassification()
  }

  def scaledClassification() {
    val spark = SparkSession.builder().appName("wine-classification2").master("local[10]").getOrCreate()
    val rawData = MLUtils.loadLibSVMFile(spark.sparkContext, "src/main/resources/data/wine.scale")
    val splittedData = rawData.randomSplit(Array(0.8, 0.2), 21)
    val trainingData = splittedData(0)
    val testData = splittedData(1)

    val classifier = DecisionTree.trainClassifier(
      input = trainingData,
      numClasses = 4,
      categoricalFeaturesInfo = Map[Int, Int](),
      impurity = "entropy",
      maxDepth = 5,
      maxBins = 32)

    val predictions = classifier.predict(testData.map(_.features))

    val labelAndPredictions = testData.map(_.label).zip(predictions)
    val mcm = new MulticlassMetrics(labelAndPredictions)
    println("Accuracy", mcm.accuracy)
    println("Precision for 1.0", mcm.precision(1.0))
    println("Precision for 2.0", mcm.precision(2.0))
    println("Precision for 3.0", mcm.precision(3.0))
    println(mcm.confusionMatrix)
    println(classifier.toDebugString)

  }

  def nonScaledClassificatio() {
    val spark = SparkSession.builder().appName("wine-classification").master("local[*]").getOrCreate()
    val rawData = spark.sparkContext.textFile("src/main/resources/data/wine.data")

    def parsePoint(line: String) = {
      val items = line.split(",")
      LabeledPoint(items(0).toInt, Vectors.dense(items.drop(1).map(_.toDouble)))
    }

    val parsedData = rawData.map(parsePoint)
    val splittedData = parsedData.randomSplit(Array(.7, .3), 21)
    val trainingData = splittedData(0)
    val testData = splittedData(1)
    val classifier = DecisionTree.trainClassifier(
      input = trainingData,
      numClasses = 4,
      categoricalFeaturesInfo = Map[Int, Int](),
      impurity = "gini",
      maxDepth = 3,
      maxBins = 32)

    val predictions = classifier.predict(testData.map(_.features))

    val labelsAndPredictions = testData.map(_.label).zip(predictions)

    val mcm = new MulticlassMetrics(labelsAndPredictions)
    println("Accuracy", mcm.accuracy)
    println("Precision for 1.0", mcm.precision(1.0))
    println("Precision for 2.0", mcm.precision(2.0))
    println("Precision for 3.0", mcm.precision(3.0))
    println(mcm.confusionMatrix)
    println(classifier.toDebugString)
  }
}