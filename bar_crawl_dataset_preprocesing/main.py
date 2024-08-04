from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, abs, when, floor


def is_inebriated(tac_reading):
    if tac_reading >= 0.08:
        return 1
    else:
        return 0


if __name__ == '__main__':
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    is_inebriated_expr = when(col("TAC_Reading") >= 0.08, 1).otherwise(0)

    BK7610 = spark.read.csv("BK7610_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("BK7610").alias("pid"))
    BU4707 = spark.read.csv("BU4707_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("BU4707").alias("pid"))
    CC6740 = spark.read.csv("CC6740_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("CC6740").alias("pid"))
    DC6359 = spark.read.csv("DC6359_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("DC6359").alias("pid"))
    DK3500 = spark.read.csv("DK3500_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("DK3500").alias("pid"))
    HV0618 = spark.read.csv("HV0618_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("HV0618").alias("pid"))
    JB3156 = spark.read.csv("JB3156_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("JB3156").alias("pid"))
    JR8022 = spark.read.csv("JR8022_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("JR8022").alias("pid"))
    MC7070 = spark.read.csv("MC7070_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("MC7070").alias("pid"))
    MJ8002 = spark.read.csv("MJ8002_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("MJ8002").alias("pid"))
    PC6771 = spark.read.csv("PC6771_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("PC6771").alias("pid"))
    SA0297 = spark.read.csv("SA0297_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("SA0297").alias("pid"))
    SF3079 = spark.read.csv("SF3079_clean_TAC.csv", header=True).select(col("timestamp").cast("long"), is_inebriated_expr.alias("is_inebriated"), lit("SF3079").alias("pid"))

    united_inebriation_df = BK7610.union(BU4707).union(CC6740).union(DC6359).union(DK3500).union(HV0618).union(JB3156).union(JR8022).union(MC7070).union(MJ8002).union(PC6771).union(SA0297).union(SF3079).withColumnRenamed("timestamp", "timestamp_2")


    data_pids = spark.read.csv("all_accelerometer_data_pids_13.csv", header=True).withColumnRenamed("time", "timestamp").withColumn("timestamp", floor(col("timestamp") / 1000)).filter(col("timestamp") != 0)

    joined_df = data_pids.join(united_inebriation_df, on=["pid"], how="left").withColumn("timestamp_diff", abs(col("timestamp") - col("timestamp_2"))).filter(col("timestamp_diff") < 10)

    final_df = joined_df.select(data_pids["*"], col("is_inebriated").alias("class")).drop("pid")

    final_df.toPandas().to_csv("bar_crawl.csv", index=False)


