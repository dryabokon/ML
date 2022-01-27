import time
import os
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, FloatType
# ----------------------------------------------------------------------------------------------------------------------
def init_spark():
    # Spark context available as 'sc' (master = local[*], app id = local-1635864339133).
    # SparkSession available as 'spark'.
    # spark = SparkSession.builder.getOrCreate()
    # sc = spark.sparkContext
    try:
        spark
    except NameError:
        spark = SparkSession.builder.getOrCreate()

    try:
        sc
    except NameError:
        sc = spark.sparkContext
    return spark,sc
# ----------------------------------------------------------------------------------------------------------------------
spark, sc = init_spark()
sqlContext = SQLContext(sc)
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def compose_path(filename_local):return os.getcwd()+filename_local
# ----------------------------------------------------------------------------------------------------------------------
def ex_hellow_world():
    RDD_nums = sc.parallelize([1, 2, 3, 4])
    values = RDD_nums.map(lambda x: x * x).collect()
    print(values)
    return values
# ----------------------------------------------------------------------------------------------------------------------
def ex_DB_read():
    CLOUDSQL_INSTANCE_IP = '104.155.188.32'  # CHANGE (database server IP)
    CLOUDSQL_DB_NAME = 'recommendation_spark'
    CLOUDSQL_USER = 'root'
    CLOUDSQL_PWD = 'easyPassword1@'  # CHANGE (root password)
    jdbcDriver = 'com.mysql.jdbc.Driver'
    jdbcUrl = 'jdbc:mysql://%s/%s?user=%s&password=%s' % (CLOUDSQL_INSTANCE_IP, CLOUDSQL_DB_NAME, CLOUDSQL_USER, CLOUDSQL_PWD)
    dfRates = sqlContext.read.format('jdbc').options(driver=jdbcDriver, url=jdbcUrl, dbtable='Rating',useSSL='false').load()
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_DB_write(allPredictions,jdbcUrl):

    schema = StructType([StructField("userId", StringType(), True), StructField("accoId", StringType(), True),StructField("prediction", FloatType(), True)])
    dfToSave = sqlContext.createDataFrame(allPredictions, schema)
    dfToSave.write.jdbc(url=jdbcUrl, table='Recommendation', mode='overwrite')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    time_start = time.time()
    ex_hellow_world()
    print('Execution time: %.2fs'%(time.time()-time_start))

