package com.aric.samples.sparkml

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector

object WineClassificationWithPreprocess extends App {

  val spark = SparkSession.builder().appName("WineClassificationWithPreprocess_APP").master("local[*]").getOrCreate()

  import spark.sqlContext.implicits._
  import spark.implicits._
  
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
    "Proline")
    
  val va = new VectorAssembler

  val vdf = va.setInputCols(Array(
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
    "Proline")).setOutputCol("features").transform(df)
    
    vdf.foreach(r=>println(r))

}