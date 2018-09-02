package com.aric.samples.sparkml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object AdultDataClassificationWithRandomForest extends App {
  val spark = SparkSession.builder().appName("AdultDataClassificationWithRandomForest").enableHiveSupport().master("local[*]").getOrCreate()

  import spark.sqlContext.implicits._

  val rawData = spark.read
    .option("header", "false")
    .option("ignoreLeadingWhiteSpace", "true")
    .csv("src/main/resources/data/adult.csv")

  val adultDataFrame = rawData.toDF(
    "Age",
    "WorkClass",
    "FnlWgt",
    "Education",
    "EducationNum",
    "MaritalStatus",
    "Occupation",
    "Relationship",
    "Race",
    "Gender",
    "CapitalGain",
    "CapitalLoss",
    "HoursPerWeek",
    "NativeCountry",
    "Label")


  val cleanDataFrame = adultDataFrame.drop($"FnlWgt") // drop unused column
    //convert ? to null and then drop na value rows
    .columns.foldLeft(adultDataFrame) {
    (df, colName) => {
      df.withColumn(colName, when(col(colName) === "?", null).otherwise(col(colName)))
    }
  }.na.drop("any")
    //cast numeric values to be numeric
    .withColumn("Age", $"Age".cast("float"))
    .withColumn("EducationNum", $"EducationNum".cast("float"))
    .withColumn("CapitalGain", $"CapitalGain".cast("float"))
    .withColumn("CapitalLoss", $"CapitalLoss".cast("float"))
    .withColumn("HoursPerWeek", $"HoursPerWeek".cast("float"))

  val categoricalFeatures = List("WorkClass",
    "Education",
    "MaritalStatus",
    "Occupation",
    "Relationship",
    "Race",
    "Gender",
    "NativeCountry")

  val featuresRequired = List("Age",
    "EducationNum",
    "CapitalGain",
    "CapitalLoss",
    "HoursPerWeek",
    "WorkClass_encoded",
    "Education_encoded",
    "MaritalStatus_encoded",
    "Occupation_encoded",
    "Relationship_encoded",
    "Race_encoded",
    "Gender_encoded",
    "NativeCountry_encoded")

  val indexers = categoricalFeatures.map(f => new StringIndexer().setInputCol(f).setOutputCol(s"${f}_index").setHandleInvalid("keep"))
  val encoderInputs = categoricalFeatures.map(f => s"${f}_index")
  val encoderOutputs = categoricalFeatures.map(f => s"${f}_encoded")
  val encoder = new OneHotEncoderEstimator().setInputCols(encoderInputs.toArray).setOutputCols(encoderOutputs.toArray)
  val labelIndexer = new StringIndexer().setInputCol("Label").setOutputCol("label_index")
  val vectorAssembler = new VectorAssembler().setInputCols(featuresRequired.toArray).setOutputCol("features")


  val stages = vectorAssembler :: encoder :: labelIndexer :: indexers

  val pipeline = new Pipeline().setStages(stages.reverse.toArray)
  val preparedDataFrame = pipeline.fit(cleanDataFrame).transform(cleanDataFrame).selectExpr("label_index as label", "features")

  val trainingAndTestData = preparedDataFrame.randomSplit(Array(0.8, 0.2), 23)

  val trainingData = trainingAndTestData(0)
  val testData = trainingAndTestData(1)

  val rfc = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label").setPredictionCol("prediction").setMaxDepth(5).setImpurity("gini").setMaxBins(35)
  val randomForestClassificationModel = rfc.fit(trainingData)
  val predictionOnTest = randomForestClassificationModel.transform(testData)
  List("f1","accuracy").foreach { m =>
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName(m)
    val evaluationResult = evaluator.evaluate(predictionOnTest)
    println(s"${evaluator.getMetricName} is ${evaluationResult}")
  }
}