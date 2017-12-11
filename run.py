from os import listdir
from os.path import isfile, join
from pyspark.sql import functions
from pyspark.sql.types import IntegerType, BooleanType
from pyspark.sql.functions import unix_timestamp, udf, col
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import DenseVector
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

import numpy as np

sqlContext = SQLContext(sc)

def readFiles(mypath):
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    result = []
    for filename in filenames:
        df = sqlContext.read.load(mypath + filename, format="com.databricks.spark.csv", header="true", inferSchema="true", delimiter=',')
        dict_name = filename.replace(".csv","")
        result.append((dict_name, df))
    return dict(result)

def extracttime(df, header, time_suffix = "_datetime"):
    df = df.withColumn(header + "_dotw", udf(lambda x: x.weekday(), IntegerType())(header + time_suffix))
    df = df.withColumn(header + "_weekend", udf(lambda x: 1 if x.weekday() == 4 or x.weekday() == 5 else 0, IntegerType())(header + time_suffix))
    df = df.withColumn(header + "_day", udf(lambda x: x.day, IntegerType())(header + time_suffix))
    df = df.withColumn(header + "_month", udf(lambda x: x.month, IntegerType())(header + time_suffix))
    df = df.withColumn(header + "_hour", udf(lambda x: x.hour, IntegerType())(header + time_suffix))
    return df


d = readFiles("data/")

# Transform date_info to month day to be joined later
d["date_info"] = extracttime(d["date_info"], "calendar", time_suffix="_date")


# One giant store_info with info from both side
d["store_info"] = d["store_id_relation"].join(d["hpg_store_info"], "hpg_store_id", "inner").join(d["air_store_info"], "air_store_id", "inner")

# Join date_info to see which days are holidays
d["air_reserve"] = extracttime(extracttime(d["air_reserve"], "reserve"), "visit")
d["air_reserve"] = d["air_reserve"].join(d["date_info"], [d["air_reserve"]["visit_day"] == d["date_info"]["calendar_day"], d["air_reserve"]["visit_month"] == d["date_info"]["calendar_month"]], "left")
d["hpg_reserve"] = extracttime(extracttime(d["hpg_reserve"], "reserve"), "visit")
d["hpg_reserve"] = d["hpg_reserve"].join(d["date_info"], [d["hpg_reserve"]["visit_day"] == d["date_info"]["calendar_day"], d["hpg_reserve"]["visit_month"] == d["date_info"]["calendar_month"]], "left")

# Break genre into numbers
hpg_genre_dict = dict(d["store_info"].select("hpg_genre_name").rdd.distinct().map(lambda r: r[0]).zipWithIndex().collect())
air_genre_dict = dict(d["store_info"].select("air_genre_name").rdd.distinct().map(lambda r: r[0]).zipWithIndex().collect())

global_hpg_genre_dict = sc.broadcast(hpg_genre_dict)
global_air_genre_dict = sc.broadcast(air_genre_dict)
hpg_genre_udf = udf(lambda x: global_hpg_genre_dict.value[x])
air_genre_udf = udf(lambda x: global_air_genre_dict.value[x])

d["store_info"] = d["store_info"].withColumn("hpg_genre_id", hpg_genre_udf("hpg_genre_name"))
d["store_info"] = d["store_info"].withColumn("air_genre_id", air_genre_udf("air_genre_name"))


# Aggregate visits for each store
d["air_visit_data"] = extracttime(d["air_visit_data"], "visit", time_suffix="_date")

air_store_visit_agg = d["air_visit_data"].groupBy("air_store_id").agg(
    functions.mean("visitors").alias("avg_no_of_visitors"),
    functions.count("visitors").alias("no_of_visits"),
    functions.sum("visitors").alias("total_no_of_visitors"),
    functions.sum(udf(lambda t: t.weekday() == 4 or t.weekday() == 5, BooleanType())("visit_date").cast("int")).alias("no_of_visits_weekend")
)

air_store_visit_extra_agg = d["air_visit_data"].groupBy(["air_store_id", "visit_weekend"]).agg(
    functions.avg("visitors").alias("avg_no_of_visitors_weekend"),
    functions.sum("visitors").alias("no_of_visitors_weekend")
)
air_store_visit_agg = air_store_visit_agg.join(air_store_visit_extra_agg, "air_store_id", "inner")


d["air_reserve"] = d["air_reserve"].join(air_store_visit_agg, ["air_store_id", "visit_weekend"], "inner")

# Input
d["input"] = d["air_reserve"].select("reserve_visitors", "visit_weekend", "visit_dotw", "visit_day", "visit_month", "visit_hour", "avg_no_of_visitors", "no_of_visits", "total_no_of_visitors", "no_of_visits_weekend", "avg_no_of_visitors_weekend", "no_of_visitors_weekend")
d["train_data"] = d["input"].rdd.map(lambda x: (x[0], DenseVector(x[1:])))
d["input_train"] = spark.createDataFrame(d["train_data"], ["label", "features"])


# Sample Submission
d["sample_submission"] = d["sample_submission"].withColumn("air_store_id", udf(lambda x: "_".join(x.split("_")[0:2]))("id"))
d["sample_submission"] =d["sample_submission"].withColumn("visit_time", udf(lambda x: x.split("_")[2])("id")).select("*", col("visit_time").cast("timestamp").alias("visit_datetime"))
d["sample_submission"] = extracttime(d["sample_submission"], "visit")
d["sample_submission"] = d["sample_submission"].join(air_store_visit_agg, ["air_store_id", "visit_weekend"], "inner")
d["test"] = d["sample_submission"].select("id", "visit_weekend", "visit_dotw", "visit_day", "visit_month", "visit_hour", "avg_no_of_visitors", "no_of_visits", "total_no_of_visitors", "no_of_visits_weekend", "avg_no_of_visitors_weekend", "no_of_visitors_weekend")
d["test_data"] = d["test"].rdd.map(lambda x: (x[0], DenseVector(x[1:])))
d["input_test"] = spark.createDataFrame(d["test_data"], ["id", "features"])





# GBT
gbt = GBTRegressor(maxIter=10)

# Hyperparameter Tuning
paramGrid = ParamGridBuilder()\
        .addGrid(gbt.subsamplingRate, np.arange(0.1, 1.1, 0.1))\
        .addGrid(gbt.stepSize, np.arange(0.1, 1.1, 0.1))\
        .build()
tvs = TrainValidationSplit(
    estimator=gbt,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(),
    trainRatio=0.8
)
model = tvs.fit(d["input_train"])

break

prediction = model.transform(d["input_test"])
prediction = prediction.withColumn("visitors", udf(lambda x: int(x) if x > 0 else 0)("prediction").cast(IntegerType())).select("id", "visitors")
prediction.toPandas().to_csv("submission.csv", index=False)
#prediction.write.csv("submission_files")
#prediction.coalesce(1).write.format("com.databricks.spark.csv").save("submission")
