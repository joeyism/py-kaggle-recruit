from os import listdir
from os.path import isfile, join
from pyspark.sql import functions
from pyspark.sql.functions import *
from pyspark.sql.types import *

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

d["store_info"] = d["store_id_relation"].join(d["hpg_store_info"], d["hpg_store_info"]["hpg_store_id"] == d["store_id_relation"]["hpg_store_id"], "inner").join(d["air_store_info"], d["air_store_info"]["air_store_id"] == d["store_id_relation"]["air_store_id"], "inner")

d["air_reserve"] = extracttime(extracttime(d["air_reserve"], "reserve"), "visit")
d["air_reserve"] = d["air_reserve"].join(d["date_info"], d["air_reserve"]["visit_datetime"] == d["date_info"]["calendar_date"], "inner")
d["hpg_reserve"] = extracttime(extracttime(d["hpg_reserve"], "reserve"), "visit")
d["hpg_reserve"] = d["hpg_reserve"].join(d["date_info"], d["hpg_reserve"]["visit_datetime"] == d["date_info"]["calendar_date"], "inner")

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
).where(col("visit_weekend") == 1)
air_store_visit_agg = air_store_visit_agg.join(air_store_visit_extra_agg, air_store_visit_agg["air_store_id"] == air_store_visit_extra_agg["air_store_id"], "inner")

