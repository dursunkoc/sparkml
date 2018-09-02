package com.aric.samples.sparkml

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Import85WitCrossValidation extends App {
  val spark = SparkSession.builder().appName("Import85WitCrossValidation").master("local[*]").enableHiveSupport().config("spark.driver.memory", "2g").getOrCreate()

  import spark.sqlContext.implicits._

  val rawData = spark.read.option("header", "true").csv("src/main/resources/data/imports-85.data")


  val dataFrame = rawData.select(
    $"price".cast("float"),
    $"make",
    $"num-of-doors",
    $"body-style",
    $"drive-wheels",
    $"wheel-base".cast("float"),
    $"curb-weight".cast("float"),
    $"num-of-cylinders",
    $"engine-size".cast("float"),
    $"horsepower".cast("float"),
    $"peak-rpm".cast("float"))

  val cleanedData = dataFrame.columns.foldLeft(dataFrame) { (df, colName) =>
    df.withColumn(colName, when(col(colName) === "?", null).otherwise(col(colName)))
  }.na.drop("any")

  val categoricalFeatures = List(
    "make",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "num-of-cylinders")

  val indexers = categoricalFeatures.map(f => new StringIndexer().setInputCol(f).setOutputCol(s"${f}_index").setHandleInvalid("keep"))

  val indexColumns = categoricalFeatures.map(f => s"${f}_index")
  val encodedColumns = categoricalFeatures.map(f => s"${f}_encoded")

  val encoder = new OneHotEncoderEstimator().setInputCols(indexColumns.toArray).setOutputCols(encodedColumns.toArray).setHandleInvalid("keep")

  val requiredColumns = Array(
    "make_encoded",
    "num-of-doors_encoded",
    "body-style_encoded",
    "drive-wheels_encoded",
    "wheel-base",
    "curb-weight",
    "num-of-cylinders_encoded",
    "engine-size",
    "horsepower",
    "peak-rpm")

  val va = new VectorAssembler().setOutputCol("features").setInputCols(requiredColumns)
  val lr = new LinearRegression().setLabelCol("price").setFeaturesCol("features").setPredictionCol("prediction").setElasticNetParam(.8).setMaxIter(50).setRegParam(1.0)

  val stages = lr :: va :: encoder :: indexers

  val pipeline = new Pipeline().setStages(stages.reverse.toArray)

  val trainAndTestData = cleanedData.randomSplit(Array(.8, .2), 21)
  val trainData = trainAndTestData(0)
  val testData = trainAndTestData(1)

  val model = pipeline.fit(trainData)

  val predictions = model.transform(testData)

  println("No Cross Validation")
  List("r2","rmse").foreach( m => {
    val regressionEvaluator = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction").setMetricName(m)
    val result = regressionEvaluator.evaluate(predictions)
    println(s"${m} is ${result}")
  })

  val paramGrid = new ParamGridBuilder()
    .addGrid(param = lr.maxIter, values = Array(10,100))
    .addGrid(param=lr.elasticNetParam, values=Array(0,0.2,0.8,1))
    .addGrid(param=lr.regParam, values=Array(0,0.1,1))
    .build()

  import org.apache.spark.ml.tuning.CrossValidator
  val r2_eval = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction").setMetricName("r2")
  val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(r2_eval).setEstimatorParamMaps(paramGrid).setParallelism(3).setNumFolds(3)
  val cvModel: CrossValidatorModel = cv.fit(trainData)
  val cvPreds = cvModel.transform(testData)

  println("Cross Validation")
  private val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
  println(s"training r2 score is ${r2_eval.evaluate(bestModel.transform(trainData))}")
  val lrModel: LinearRegressionModel = bestModel.stages.last.asInstanceOf[LinearRegressionModel]
  val hyperParams = lrModel.params.map(p=>(p.name, lrModel.get(p)))
  hyperParams.foreach(println)
//  lrModel.get(lrModel.params(1))

  List("r2","rmse").foreach( m => {
    val regressionEvaluator = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction").setMetricName(m)
    val result = regressionEvaluator.evaluate(cvPreds)
    println(s"${m} is ${result}")
  })
}
