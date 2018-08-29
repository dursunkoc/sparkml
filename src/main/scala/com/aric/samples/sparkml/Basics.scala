package com.aric.samples.sparkml

import org.apache.spark.ml.linalg.{ Matrix, Vectors }
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

object Basics extends App {

  val spark = SparkSession.builder().appName("basics").master("local[*]").getOrCreate()
  val sqlContext = spark.sqlContext
  import sqlContext.implicits._

  val data = Seq(
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0))))

  val df = data.map(Tuple1.apply).toDF("features")

  val coeff1 = Correlation.corr(df, "features").head
  println(s"Pearson correlation matrix:\n $coeff1")
  
  val coeff2 = Correlation.corr(df, "features", "spearman").head
  println(s"Pearson correlation matrix:\n $coeff2")

}