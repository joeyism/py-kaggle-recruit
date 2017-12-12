from os import listdir
from os.path import isfile, join
from pyspark import SparkConf, SparkContext
from pyspark.sql import functions, SQLContext
from pyspark.sql.types import IntegerType, BooleanType, DateType
from pyspark.sql.functions import unix_timestamp, udf, col, lit
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import DenseVector
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from datetime import datetime
import numpy as np

#TODO: Use hpg data, since air -> store_info -> hpg
#TODO: find day_since_01_01 for each year

if sc == None:
    sc = SparkContext(appName="recruit")

if spark == None:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder\
            .master("local")\
            .appName("Word Count")\
            .config("spark.some.config.option", "some-value")\
            .getOrCreate()



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
    df = df.withColumn(header + "_year", udf(lambda x: x.year, IntegerType())(header + time_suffix))
    df = df.withColumn(header + "_hour", udf(lambda x: x.hour, IntegerType())(header + time_suffix))
    df = df.withColumn(header + "_days_since_newyear", udf(lambda x: (x - datetime.strptime(str(x.year) + "-01-01", "%Y-%m-%d")).days, IntegerType())(header + time_suffix))
    return df

def get_feature_importance_dict(feature, score):
    result = []
    for i in range(len(feature)):
        result.append((score[i], feature[i]))
    return dict(result)


d = readFiles("data/")

# Transform date_info to month day to be joined later
d["date_info"] = extracttime(d["date_info"], "calendar", time_suffix="_date")


# One giant store_info with info from both side
d["store_info"] = d["store_id_relation"].join(d["hpg_store_info"], "hpg_store_id", "inner").join(d["air_store_info"], "air_store_id", "inner")

# Join date_info to see which days are holidays
d["air_reserve"] = extracttime(extracttime(d["air_reserve"], "reserve"), "visit")
d["air_reserve"] = d["air_reserve"].join(d["date_info"], [d["air_reserve"]["visit_day"] == d["date_info"]["calendar_day"], d["air_reserve"]["visit_month"] == d["date_info"]["calendar_month"], d["air_reserve"]["visit_year"] == d["date_info"]["calendar_year"]], "left")
d["hpg_reserve"] = extracttime(extracttime(d["hpg_reserve"], "reserve"), "visit")
d["hpg_reserve"] = d["hpg_reserve"].join(d["date_info"], [d["hpg_reserve"]["visit_day"] == d["date_info"]["calendar_day"], d["hpg_reserve"]["visit_month"] == d["date_info"]["calendar_month"], d["hpg_reserve"]["visit_year"] == d["date_info"]["calendar_year"]], "left")

# Break genre into numbers
hpg_genre_dict = dict(d["store_info"].select("hpg_genre_name").rdd.distinct().map(lambda r: r[0]).zipWithIndex().collect())
air_genre_dict = dict(d["store_info"].select("air_genre_name").rdd.distinct().map(lambda r: r[0]).zipWithIndex().collect())

global_hpg_genre_dict = sc.broadcast(hpg_genre_dict)
global_air_genre_dict = sc.broadcast(air_genre_dict)
hpg_genre_udf = udf(lambda x: global_hpg_genre_dict.value[x])
air_genre_udf = udf(lambda x: global_air_genre_dict.value[x])

d["store_info"] = d["store_info"].withColumn("hpg_genre_id", hpg_genre_udf("hpg_genre_name").cast(IntegerType()))
d["store_info"] = d["store_info"].withColumn("air_genre_id", air_genre_udf("air_genre_name").cast(IntegerType()))

# Break area into numbers
air_area_dict = dict(d["store_info"].select("air_area_name").rdd.distinct().map(lambda r: r[0]).zipWithIndex().collect())
global_air_area_dict = sc.broadcast(air_area_dict)
air_area_udf = udf(lambda x: global_air_area_dict.value[x])
d["store_info"] = d["store_info"].withColumn("air_area_id", air_area_udf("air_area_name").cast(IntegerType()))

# breaks air_store_id into unique num value
air_store_id_dict = dict(d["store_info"].select("air_store_id").rdd.distinct().map(lambda r: r[0]).zipWithIndex().collect())
global_air_store_id_dict  = sc.broadcast(air_store_id_dict )
air_store_udf = udf(lambda x:global_air_store_id_dict.value[x])
d["store_info"] = d["store_info"].withColumn("air_store_num",air_store_udf("air_store_id").cast(IntegerType()))



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

# Join with aggregates
d["air_reserve"] = d["air_reserve"].join(air_store_visit_agg, ["air_store_id", "visit_weekend"], "inner").join(d["store_info"], "air_store_id", "left")



##### Getting train and test data
#
#
# Input
features_col = ["visit_dotw", "visit_day", "visit_month", "avg_no_of_visitors", "no_of_visits", "total_no_of_visitors", "no_of_visits_weekend", "avg_no_of_visitors_weekend", "no_of_visitors_weekend", "air_area_id", "visit_days_since_newyear"]

d["input"] = d["air_reserve"].select(np.append(["reserve_visitors"], features_col).tolist() )
d["train_data"] = d["input"].rdd.map(lambda x: (x[0], DenseVector(x[1:])))
d["input_train"] = spark.createDataFrame(d["train_data"], ["label", "features"])


# Sample Submission
d["sample_submission"] = d["sample_submission"].withColumn("air_store_id", udf(lambda x: "_".join(x.split("_")[0:2]))("id"))
d["sample_submission"] =d["sample_submission"].withColumn("visit_time", udf(lambda x: x.split("_")[2])("id")).select("*", col("visit_time").cast("timestamp").alias("visit_datetime"))
d["sample_submission"] = extracttime(d["sample_submission"], "visit")
d["sample_submission"] = d["sample_submission"]\
        .join(air_store_visit_agg, ["air_store_id", "visit_weekend"], "inner")\
        .join(d["store_info"], "air_store_id", "left")\
        .join(d["date_info"], ((d["sample_submission"]["visit_day"] == d["date_info"]["calendar_day"]) & (d["sample_submission"]["visit_month"] == d["date_info"]["calendar_month"]) & (d["sample_submission"]["visit_year"] == d["date_info"]["calendar_year"])), "inner")

d["test"] = d["sample_submission"].select(np.append(["id"],features_col).tolist() )
d["test_data"] = d["test"].rdd.map(lambda x: (x[0], DenseVector(x[1:])))
d["input_test"] = spark.createDataFrame(d["test_data"], ["id", "features"])





# GBT
gbt = GBTRegressor(maxIter=10)\
        .setSubsamplingRate(0.9)\
        .setStepSize(0.4)


# Hyperparameter Tuning
# paramGrid = ParamGridBuilder()\
#         .addGrid(gbt.subsamplingRate, np.arange(0.1, 1.1, 0.1))\
#         .addGrid(gbt.stepSize, np.arange(0.1, 1.1, 0.1))\
#         .build()
# tvs = TrainValidationSplit(
#     estimator=gbt,
#     estimatorParamMaps=paramGrid,
#     evaluator=RegressionEvaluator(),
#     trainRatio=0.8
# )
# model = tvs.fit(d["input_train"])
# 
# print("Optimal Subsampling Rate: {}".format(model.bestModel._java_obj.getSubsamplingRate()))
# print("Optimal Step size: {}".format(model.bestModel._java_obj.getStepSize()))


model = gbt.fit(d["input_train"])

feature_importance_dict = get_feature_importance_dict(features_col, model.featureImportances)

prediction = model.transform(d["input_test"])
prediction = prediction.withColumn("visitors", udf(lambda x: int(x) if x > 0 else 0)("prediction").cast(IntegerType())).select("id", "visitors")
prediction.toPandas().to_csv("submission.csv", index=False)
#prediction.write.csv("submission_files")
#prediction.coalesce(1).write.format("com.databricks.spark.csv").save("submission")
